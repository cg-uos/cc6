"""
raycaster_cc6.py

# Copyright (c) 2022, Minho Kim & Hyunjun Kim
# Computer Graphics Lab, Dept of Computer Science, University of Seoul
# All rights reserved.

"""
import sys
from OpenGL.GL import *
import collections
import pylab
import time
import numpy as np
import glm
import os
import platform
import glfw # https://github.com/FlorianRhiem/pyGLFW

class VolumeInfo:
    def __init__(self, filename, dtype, dim, scale, level):
        self.filename = filename
        self.dtype = dtype
        self.dim = dim
        self.scale = scale
        self.level = level
        self.offset = (0,0,0)
        self.bbox_size = tuple((dim[i]-1)*scale[i] for i in range(3))

class ShaderInfo:
    def __init__(self, name, prog):
        self.name = name
        self.prog = prog

path_volume = './'

volumes = {
    'ML40' :VolumeInfo(path_volume + 'ML_40_lattice.raw', np.float32, (41,41,41), (1,1,1), 0.5),
    'ML50' :VolumeInfo(path_volume + 'ML_50_lattice.raw', np.float32, (51,51,51), (1,1,1), 0.5),
    'ML80' :VolumeInfo(path_volume + 'ML_80_lattice.raw', np.float32, (81,81,81), (1,1,1), 0.5),
    }

#############################################################################################################
class BBox:
    def __init__(self, bbox_size, size_fbo):
        self.fbo = FBO_bbox(size_fbo[0], size_fbo[1])
        self.prog_bbox = Program('bbox.vert', 'bbox.frag', ['MVP', 'scale'])  
        size_max = max(bbox_size)
        self.scale_bbox_norm = tuple(bbox_size[i]/size_max for i in range(3))
        positions = np.array([  0, 0, 1,
                                1, 0, 1,
                                1, 1, 1,
                                0, 1, 1,
                                0, 0, 0,
                                1, 0, 0,
                                1, 1, 0,
                                0, 1, 0],
                                dtype=np.float32)
        indices = np.array([    0, 1, 2, 2, 3, 0, # front
                                1, 5, 6, 6, 2, 1, # top
                                7, 6, 5, 5, 4, 7, # back
                                4, 0, 3, 3, 7, 4, # bottom
                                4, 5, 1, 1, 0, 4, # left
                                3, 2, 6, 6, 7, 3 # right
                                ], dtype=np.int8)
 
        # Setting up the VAO for the bbox
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo_position = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_position)
        glBufferData(GL_ARRAY_BUFFER, len(positions)*ctypes.sizeof(ctypes.c_float), positions, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.vbo_idx = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_idx)
        self.size_indices = len(indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices)*ctypes.sizeof(ctypes.c_ubyte), indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def render(self, MVP):
        glUseProgram(self.prog_bbox.id)
        glUniformMatrix4fv(self.prog_bbox.uniform_locs['MVP'], 1, GL_FALSE, MVP)
        glUniform3fv(self.prog_bbox.uniform_locs['scale'], 1, self.scale_bbox_norm)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.size_indices, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glBindVertexArray(0)
        glUseProgram(0)

    def render_backfaces(self, MVP):
        glDepthFunc(GL_GREATER)
        glClearDepth(0)
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        self.render(MVP)
        glDisable(GL_CULL_FACE)

    def render_frontfaces(self, MVP):
        glDepthFunc(GL_LESS)
        glClearDepth(1)
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        self.render(MVP)
        glDisable(GL_CULL_FACE)

    def render_bbox(self, MVP):
        glViewport(0, 0, self.fbo.width, self.fbo.height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.fbo.buf_back, 0)
        self.render_backfaces(MVP)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.fbo.buf_front, 0)
        self.render_frontfaces(MVP)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

#############################################################################################################
class Volume:
    def __init__(self, info, size_fbo_bbox):
        self.load_data(info)
        self.bbox = BBox(self.info.bbox_size, size_fbo_bbox)
        bbox_size_max = max(self.info.bbox_size)
        self.scale_norm = tuple(self.info.scale[i]/bbox_size_max for i in range(3))
        self.offset_norm = tuple(self.info.offset[i]/bbox_size_max for i in range(3))
        self.scale_lattice =  tuple(info.bbox_size[i]/info.scale[i] for i in range(3))
        self.offset_lattice = tuple(-info.offset[i]/info.scale[i] for i in range(3))
        self.upload_data()
       
    def load_data(self, info):
        self.info = info
        self.dim_max = max(max(self.info.dim[0], self.info.dim[1]), self.info.dim[2])
        self.dim_tex = [self.info.dim[0], self.info.dim[1], self.info.dim[2], 1]
        self.data = np.fromfile(info.filename, dtype=info.dtype).astype(np.float32)

    def upload_data(self):
        self.texid = glGenTextures(1)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glBindTexture(GL_TEXTURE_3D, self.texid)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, self.dim_tex[0], self.dim_tex[1], self.dim_tex[2], 0, GL_RED, GL_FLOAT, self.data)
        glBindTexture(GL_TEXTURE_3D, 0)
        self.data = None

#############################################################################################################
class FBO_bbox:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        self.buf_back = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.buf_back)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.buf_front = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.buf_front)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.rbo)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

#############################################################################################################
class QuadFull:
    def __init__(self, volume, size_fbo):
        self.tex_bbox_back = volume.bbox.fbo.buf_back
        self.tex_bbox_front = volume.bbox.fbo.buf_front
        self.tex_volume = volume.texid
        self.init_colormap()

        try:
            self.progs = {}
            uniforms = ['dim','level', 'tex_colormap', 'tex_colormap_2d', 'tex_back', 'tex_front', 'MV', 'scale_lattice', 'offset_lattice', 'tex_volume', 'scale_norm', 'offset_norm', 'dim_max', 'scale_step', 'scale_delta' ]
            self.progs['cc6'] = ShaderInfo('cc6-principal-curvature', Program('raycast_simple.vert', 'cc6_raycast_curvature.frag', uniforms))
 
    
        except BaseException as err:
            print(f'Exception while compiling shaders...!: {err}')
            quit()

        self.init_vao()

        self.scale_delta = 0.01
        self.scale_step = 0.001

    def init_vao(self):
        verts = np.array(
            [-1, -1, 0, 0,
              1, -1, 1, 0,
              1,  1, 1, 1,
             -1,  1, 0, 1], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, len(verts)*ctypes.sizeof(ctypes.c_float), verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*ctypes.sizeof(ctypes.c_float), None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(2*ctypes.sizeof(ctypes.c_float)))
        glBindVertexArray(0)

    def render_raycast(self, level, volume, MV):

        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex_bbox_back)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.tex_bbox_front)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_3D, self.tex_volume)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, self.tex_colormap2d)


        prog = self.progs["cc6"].prog
        glUseProgram(prog.id)

        glUniform1i(prog.uniform_locs['tex_back'], 0)   
        glUniform1i(prog.uniform_locs['tex_front'], 1) 
        glUniform1i(prog.uniform_locs['tex_volume'], 2)
        glUniform1i(prog.uniform_locs['tex_colormap_2d'], 4)
        glUniform1f(prog.uniform_locs['level'], level)
        glUniform3f(prog.uniform_locs['scale_norm'], volume.scale_norm[0], volume.scale_norm[1], volume.scale_norm[2])
        glUniform3f(prog.uniform_locs['offset_norm'], volume.offset_norm[0], volume.offset_norm[1], volume.offset_norm[2])
        glUniform3f(prog.uniform_locs['scale_lattice'], volume.scale_lattice[0], volume.scale_lattice[1], volume.scale_lattice[2])
        glUniform3f(prog.uniform_locs['offset_lattice'], volume.offset_lattice[0], volume.offset_lattice[1], volume.offset_lattice[2])
        glUniform3f(prog.uniform_locs['dim'], volume.info.dim[0], volume.info.dim[1], volume.info.dim[2])
        glUniform1f(prog.uniform_locs['dim_max'], volume.dim_max)
        glUniformMatrix4fv(prog.uniform_locs['MV'], 1, GL_FALSE, MV)
        glUniform1f(prog.uniform_locs['scale_step'], self.scale_step)
        glUniform1f(prog.uniform_locs['scale_delta'], self.scale_delta)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)

    def init_colormap(self):
# colormap for principal curvatures courtesy of 
# G. Kindlmann, R. Whitaker, T. Tasdizen, T. Möller, 
# "Curvature-based transfer functions for direct volume rendering: Methods and applications"
# in Proceedings of IEEE Visualization 2003, 2003, pp. 513–520. 
# https://dx.doi.org/10.1109/VISUAL.2003.1250414
        colormap2d = np.array([[1,0,0], [1,1,0], [0,1,0],
                                [.5,.5,.5], [.5,.5,.5], [0,1,1],
                                [.5,.5,.5], [.5,.5,.5], [0,0,1]], dtype=np.float32)

        self.tex_colormap2d = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_colormap2d)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3, 3, 0, GL_RGB, GL_FLOAT, colormap2d)


#############################################################################################################
class Program:
    def __init__(self, filename_vert, filename_frag, uniforms):
        self.path = './'
        self.filename_vert = filename_vert
        self.filename_frag = filename_frag
        src_vert = self.load_source(filename_vert)
        src_frag = self.load_source(filename_frag)
        self.id = self.build(src_vert, src_frag, uniforms)

    def load_source(self, filename):
        # If the shader source is composed of several files, merge them.
        src = open(self.path + filename, 'r').read()
        return src

    def compile(self, src, type):
            
        id = glCreateShader(type)
        glShaderSource(id, src)
        glCompileShader(id)
        result = glGetShaderiv(id, GL_COMPILE_STATUS)

        if not(result):
            print('shader compilation error.')
            print(glGetShaderInfoLog(id))
            raise RuntimeError(
                """Shader compile failure (%s): %s"""%(
                    result,
                    glGetShaderInfoLog( id ),
                    ),
                src,
                type,
                )
        return id

    def build(self, src_vert, src_frag, uniforms):
        try:
            id_vert = self.compile(src_vert, GL_VERTEX_SHADER)
        except RuntimeError as err:
            print(f'Error while compiling {self.filename_vert}')
            raise err
        try:
            id_frag = self.compile(src_frag, GL_FRAGMENT_SHADER)
        except RuntimeError as err:
            print(f'Error while compiling {self.filename_frag}')
            raise err
        program = glCreateProgram()
        if not program:
            raise RunTimeError('glCreateProgram faled!')
    
        glAttachShader(program, id_vert)
        glAttachShader(program, id_frag)
        glLinkProgram(program)
        status = glGetProgramiv(program, GL_LINK_STATUS)
        if not status:
            infoLog = glGetProgramInfoLog(program)
            glDeleteProgram(program)
            glDeleteShader(id_vert)
            glDeleteShader(id_frag)
            print(infoLog)
            raise RuntimeError("Error linking program:\n%s\n", infoLog)

        self.uniform_locs = {}
        for u in uniforms:
            self.uniform_locs[u] = glGetUniformLocation(program, u)
        return program

#############################################################################################################
class Scene:    
    """ OpenGL 3D scene class"""

    def __init__(self, width, height):

        self.platform = platform.system()
        self.width = width
        self.height = height

        self.view_angle = 21
        self.angle_x = 320
        self.angle_y = 0
        self.position_x = 0
        self.position_y = 0

        self.volume = Volume(volumes[sys.argv[1]], (width,height))
        self.quad_full = QuadFull(self.volume, (width,height))
        self.refresh_MVP()
        self.texid = [self.volume.bbox.fbo.buf_front, self.volume.bbox.fbo.buf_back]
        self.level = volumes[sys.argv[1]].level

    def refresh_MVP(self):
        self.P = glm.perspective(glm.radians(self.view_angle), self.width/self.height, 1, 3)
        self.MV = glm.translate(glm.mat4(), glm.vec3(self.position_x, self.position_y, -2))
        self.MV = glm.rotate(self.MV, glm.radians(self.angle_x), glm.vec3(1,0,0))
        self.MV = glm.rotate(self.MV, glm.radians(self.angle_y), glm.vec3(0,1,0))
        self.MVP = np.array(glm.transpose(self.P * self.MV))
        self.MV = np.array(glm.transpose(self.MV))

    def render(self):
        self.volume.bbox.render_bbox(self.MVP)
        self.quad_full.render_raycast(self.level, self.volume, self.MV) 
#############################################################################################################
class RenderWindow:
    def __init__(self):
        cwd = os.getcwd() # save current working directory
        glfw.init() # initialize glfw - this changes cwd
        os.chdir(cwd) # restore cwd

        # version hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
   
        # make a window
        self.width, self.height = 512, 512
        self.aspect = self.width/float(self.height)
        self.win = glfw.create_window(self.width, self.height, 'raycaster', None, None)
        # make context current
        glfw.make_context_current(self.win)


        # for retina display...
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(self.win)
        print(f'glfw.get_framebuffer_size returned {(self.fb_width,self.fb_height)}')


        print("OpenGL version = ", glGetString( GL_VERSION ))
        print("GLSL version = ", glGetString( GL_SHADING_LANGUAGE_VERSION ))
        
        # initialize GL
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0,0.0)

        # set window callbacks
        glfw.set_key_callback(self.win, self.onKeyboard)
        glfw.set_window_size_callback(self.win, self.onSize)        

        # create 3D
        self.scene = Scene(self.fb_width, self.fb_height)

        # exit flag
        self.exitNow = False
        
    def onKeyboard(self, win, key, scancode, action, mods):
        def set_step_level(mods):
            if mods & glfw.MOD_SHIFT:
                step_level = 0.01
            elif mods & glfw.MOD_CONTROL:
                step_level = 0.0001
            else:
                step_level = 0.001
            return step_level


        if action == glfw.PRESS:
            # ESC to quit
            if key == glfw.KEY_ESCAPE: 
                self.exitNow = True
            # R arrow
            elif key == glfw.KEY_RIGHT:
                if mods & glfw.MOD_SHIFT:
                    if mods & glfw.MOD_CONTROL:
                        step = .01
                    else:
                        step = .1
                    self.scene.position_x += step;
                else:
                    self.scene.angle_y = (self.scene.angle_y + 10) % 360
                self.scene.refresh_MVP()
            # L arrow
            elif key == glfw.KEY_LEFT:
                if mods & glfw.MOD_SHIFT:
                    if mods & glfw.MOD_CONTROL:
                        step = .01
                    else:
                        step = .1
                    self.scene.position_x -= step;
                else:
                    self.scene.angle_y = (self.scene.angle_y - 10) % 360
                self.scene.refresh_MVP()
            # U arrow
            elif key == glfw.KEY_UP:
                if mods & glfw.MOD_SHIFT:
                    if mods & glfw.MOD_CONTROL:
                        step = .01
                    else:
                        step = .1
                    self.scene.position_y += step;
                else:
                    self.scene.angle_x = (self.scene.angle_x - 10) % 360
                self.scene.refresh_MVP()
            # D arrow
            elif key == glfw.KEY_DOWN:
                if mods & glfw.MOD_SHIFT:
                    if mods & glfw.MOD_CONTROL:
                        step = .01
                    else:
                        step = .1
                    self.scene.position_y -= step;
                else:
                    self.scene.angle_x = (self.scene.angle_x + 10) % 360
                self.scene.refresh_MVP()
            # =(+): increase the isolevel value
            elif key == glfw.KEY_EQUAL:
                if mods & glfw.MOD_SHIFT:
                    self.scene.level = self.scene.level + set_step_level(mods)*10
                else:
                    self.scene.level = self.scene.level + set_step_level(mods)
                print(self.scene.level)
            # -: decrease the isolevel value
            elif key == glfw.KEY_MINUS:
                if mods & glfw.MOD_SHIFT:
                    self.scene.level = self.scene.level - set_step_level(mods)*10
                else:
                    self.scene.level = self.scene.level - set_step_level(mods)
                print(self.scene.level)
            # PgUp: zoom in
            elif key == glfw.KEY_PAGE_UP:
                self.scene.view_angle = self.scene.view_angle - 1
                self.scene.refresh_MVP()
            # PgDn: zoom out
            elif key == glfw.KEY_PAGE_DOWN:
                self.scene.view_angle = self.scene.view_angle + 1
                self.scene.refresh_MVP()
        
    def onSize(self, win, width, height):
        self.aspect = width/float(height)
        self.scene.width = width
        self.scene.height = height

    def run(self):
        glfw.set_time(0)
        glClearColor(1,1,1,1)
        lastT = glfw.get_time()
        frames = 0
        while not glfw.window_should_close(self.win) and not self.exitNow:
            currT = glfw.get_time()
            if frames == 20:
                elapsed = currT - lastT
                print('fps = {}'.format(frames/elapsed))
                lastT = currT
                frames = 0
            self.scene.render()
            frames += 1
            glfw.swap_buffers(self.win)
            glfw.poll_events()
        glfw.terminate()

#############################################################################################################
# main() function
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python raycaster_cc6.py <volume data name>")
        print(f"\t<voluem data name>: one of {list(volumes.keys())}")
        exit(-1)
    if not sys.argv[1] in volumes:
        print(f"The 1nd argument should be one of {list(volumes.keys())}.")
        exit(-1)

    rw = RenderWindow()
    print("Starting raycaster. "
          "Press ESC to quit.")
    rw.run()

# call main
if __name__ == '__main__':
    main()
