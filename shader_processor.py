"""
Liche - GLSL Shader Processor
Singleton: lazy moderngl context, feedback texture, shader chain.
Pre-compiles all shaders at init to work around driver limitations
with multiple program creation.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _log(msg: str) -> None:
    try:
        print(msg, file=sys.stderr, flush=True)
    except OSError:
        pass


_QUAD_VERTS = np.array([
    -1.0, -1.0, 0.0, 0.0,
     1.0, -1.0, 1.0, 0.0,
    -1.0,  1.0, 0.0, 1.0,
     1.0,  1.0, 1.0, 1.0,
], dtype="f4")


class ShaderProcessor:
    """Singleton GLSL shader processor with lazy moderngl context and feedback texture."""

    _instance: Optional["ShaderProcessor"] = None

    def __new__(cls) -> "ShaderProcessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._ctx = None
        self._programs: Dict[str, Any] = {}
        self._fbo = None
        self._tex_in = None
        self._tex_out = None
        self._tex_mask = None
        self._tex_feedback = None
        self._feedback_prev: Optional[np.ndarray] = None
        self._quad_vbo = None
        self._vaos: Dict[str, Any] = {}
        self._shaders_dir = os.path.join(os.path.dirname(__file__), "SHADERS")
        self._initialized = True
        self._ready = False
        self._moderngl = None
        self._init_context()

    def _release_all(self) -> None:
        """Release all GL objects (programs, VAOs, textures, FBO) without touching the context."""
        for vao in self._vaos.values():
            if vao:
                try:
                    vao.release()
                except Exception:
                    pass
        self._vaos.clear()
        for prog in self._programs.values():
            if prog:
                try:
                    prog.release()
                except Exception:
                    pass
        self._programs.clear()
        for obj in (self._fbo, self._tex_in, self._tex_out, self._tex_mask, self._tex_feedback, self._quad_vbo):
            if obj:
                try:
                    obj.release()
                except Exception:
                    pass
        self._fbo = self._tex_in = self._tex_out = self._tex_mask = self._tex_feedback = self._quad_vbo = None
        self._feedback_prev = None

    def _init_context(self) -> None:
        try:
            import moderngl
            self._moderngl = moderngl
            self._ctx = moderngl.create_standalone_context()
            self._quad_vbo = self._ctx.buffer(_QUAD_VERTS.tobytes())
            self._ready = True
            info = self._ctx.info
            renderer = info.get("GL_RENDERER", "unknown")
            version = info.get("GL_VERSION", "unknown")
            vendor = info.get("GL_VENDOR", "unknown")
            _log(f"[Liche] ShaderProcessor ready  |  {vendor} / {renderer} / GL {version}")
            self._precompile_all_shaders()
        except Exception as e:
            _log(f"[Liche] ShaderProcessor unavailable (no GPU/modernGL): {e}")
            self._ready = False
            self._moderngl = None

    def _precompile_all_shaders(self) -> None:
        """Compile every .frag and create its VAO immediately."""
        vert_path = os.path.join(self._shaders_dir, "vertex_quad.glsl")
        if not os.path.isfile(vert_path):
            _log("[Liche] vertex_quad.glsl not found, skipping precompile")
            return
        with open(vert_path) as f:
            vert_src = f.read()

        frag_files = glob.glob(os.path.join(self._shaders_dir, "*.frag"))
        compiled = 0
        for frag_path in sorted(frag_files):
            name = os.path.splitext(os.path.basename(frag_path))[0]
            if name == "template":
                continue
            with open(frag_path) as f:
                frag_src = f.read()
            try:
                prog = self._ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
                self._programs[name] = prog
                vao = self._ctx.vertex_array(
                    prog,
                    [(self._quad_vbo, "2f 2f", "in_pos", "in_uv")],
                )
                self._vaos[name] = vao
                compiled += 1
            except Exception as e:
                _log(f"[Liche] Precompile failed for '{name}': {e}")
                self._programs[name] = None

        _log(f"[Liche] Precompiled {compiled} shaders + VAOs")

    def _reinit_context(self) -> bool:
        """Tear down and rebuild the entire GL context. Returns True on success."""
        _log("[Liche] Reinitializing GL context...")
        self._release_all()
        if self._ctx:
            try:
                self._ctx.release()
            except Exception:
                pass
        self._ctx = None
        self._ready = False
        self._init_context()
        return self._ready

    def reset_feedback(self) -> None:
        """Clear feedback texture and force texture recreation for next generation."""
        self._feedback_prev = None
        for obj in (self._fbo, self._tex_in, self._tex_out, self._tex_mask, self._tex_feedback):
            if obj:
                try:
                    obj.release()
                except Exception:
                    pass
        self._fbo = self._tex_in = self._tex_out = self._tex_mask = self._tex_feedback = None

    def _get_program(self, frag_name: str) -> Any:
        return self._programs.get(frag_name)

    def _ensure_textures(self, h: int, w: int) -> None:
        if self._ctx is None:
            return
        need_recreate = (
            self._tex_in is None
            or self._tex_in.size != (w, h)
        )
        if need_recreate:
            for obj in (self._fbo, self._tex_in, self._tex_out, self._tex_mask, self._tex_feedback):
                if obj:
                    obj.release()
            self._tex_in = self._ctx.texture((w, h), 4, dtype="f1")
            self._tex_out = self._ctx.texture((w, h), 4, dtype="f1")
            self._tex_mask = self._ctx.texture((w, h), 1, dtype="f1")
            self._tex_feedback = self._ctx.texture((w, h), 4, dtype="f1")
            self._fbo = self._ctx.framebuffer(color_attachments=[self._tex_out])
            self._feedback_prev = None

    def _get_shader_intensity_params(self, params: Any) -> List[Tuple[str, int]]:
        """Auto-detect GlitchParams fields shader_*_intensity, return (name, value) in order."""
        out: List[Tuple[str, int]] = []
        pat = re.compile(r"^shader_(.+)_intensity$")
        for name in sorted(getattr(params, "__dataclass_fields__", {}).keys()):
            m = pat.match(name)
            if m:
                val = getattr(params, name, 0)
                if isinstance(val, (int, float)) and val > 0:
                    shader_name = m.group(1)
                    out.append((shader_name, int(val)))
        return out

    def apply_shaders_chain(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        params: Any,
        frame_idx: int,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Apply all active shader_*_intensity effects in order.
        Returns frame as np.ndarray (BGR uint8). Passthrough if no GPU or no shaders.
        """
        if not self._ready or self._ctx is None:
            return frame
        chain = self._get_shader_intensity_params(params)
        if not chain:
            return frame

        if frame_idx == 0:
            _log(f"[Liche] Shader chain: {[(n, i) for n, i in chain]}")

        try:
            return self._apply_shaders_chain_impl(frame, mask, params, frame_idx, seed, chain)
        except Exception as e:
            _log(f"[Liche] Shader chain error: {e}")
            if self._reinit_context():
                try:
                    return self._apply_shaders_chain_impl(frame, mask, params, frame_idx, seed, chain)
                except Exception as e2:
                    _log(f"[Liche] Shader chain failed after reinit: {e2}")
            return frame

    def _apply_shaders_chain_impl(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        params: Any,
        frame_idx: int,
        seed: int,
        chain: List[Tuple[str, int]],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1].copy()
        if mask.ndim == 3:
            mask_1ch = mask[:, :, 0]
        else:
            mask_1ch = mask
        if mask_1ch.dtype != np.float32:
            mask_1ch = mask_1ch.astype(np.float32) / 255.0
        else:
            mask_1ch = np.clip(mask_1ch, 0, 1).astype(np.float32)

        self._ensure_textures(h, w)
        if self._fbo is None:
            return frame

        time_val = frame_idx * 0.1 + seed * 0.01
        current = rgb.astype(np.float32) / 255.0
        feedback_data = self._feedback_prev

        for shader_name, intensity in chain:
            prog = self._get_program(shader_name)
            if prog is None:
                continue
            vao = self._vaos.get(shader_name)
            if vao is None:
                continue

            # Clear any stale GL errors
            try:
                self._ctx.error
            except Exception:
                pass

            # Upload frame data
            rgba = np.dstack([np.clip(current, 0, 1), np.ones((h, w), dtype=np.float32)]) * 255
            self._tex_in.write(rgba.astype(np.uint8).tobytes())
            self._tex_mask.write((np.clip(mask_1ch, 0, 1) * 255).astype(np.uint8).tobytes())
            fb_src = feedback_data if feedback_data is not None else current
            fb_rgba = np.dstack([np.clip(fb_src, 0, 1), np.ones((h, w), dtype=np.float32)]) * 255
            self._tex_feedback.write(fb_rgba.astype(np.uint8).tobytes())

            # Set uniforms
            for uname, uval in [
                ("u_texture", 0), ("u_mask", 1), ("u_feedback", 2),
                ("u_resolution", (float(w), float(h))),
                ("u_time", time_val), ("u_frame_idx", frame_idx),
                ("u_seed", seed), ("u_intensity", float(intensity) / 10.0),
            ]:
                try:
                    prog[uname].value = uval
                except KeyError:
                    pass

            # Bind FBO, then textures, then render
            self._fbo.use()
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            self._tex_in.use(0)
            self._tex_mask.use(1)
            self._tex_feedback.use(2)
            vao.render(self._moderngl.TRIANGLE_STRIP)

            data = self._fbo.read(attachment=0, components=4, dtype="f1")
            current = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3].astype(np.float32) / 255.0
            feedback_data = current.copy()

        self._feedback_prev = feedback_data
        out = (np.clip(current, 0, 1) * 255).astype(np.uint8)
        out = out[:, :, ::-1]
        return out


def _get_or_create_processor() -> ShaderProcessor:
    """Survive Streamlit module reimports by pinning the instance to sys."""
    key = "_liche_shader_processor"
    existing = getattr(sys, key, None)
    if existing is not None and isinstance(existing, ShaderProcessor):
        return existing
    inst = ShaderProcessor()
    setattr(sys, key, inst)
    return inst


shader_processor: ShaderProcessor = _get_or_create_processor()
