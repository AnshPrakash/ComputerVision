import pygame
from OpenGL.GL import *


def MTL(filename):
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise(ValueError, "mtl file doesn't start with newmtl stmt")
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            surf = pygame.image.load(mtl['map_Kd'])
            image = pygame.image.tostring(surf, 'RGBA', 1)
            ix, iy = surf.get_rect().size
            texid = mtl['texture_Kd'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, image)
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents

class OBJ:
  def __init__(self, filename, swapyz=False):
    """Loads a Wavefront OBJ file. """
    self.vertices = []
    self.normals = []
    self.texcoords = []
    self.faces = []
    material = None
    for line in open(filename, "r"):
      if line.startswith('#'): continue
      values = line.split()
      if not values: continue
      if values[0] == 'v':
        v = list(map(float, values[1:4]))
        if swapyz:
          v = v[0], v[2], v[1]
        self.vertices.append(v)
      elif values[0] == 'vn':
        v = list(map(float, values[1:4]))
        if swapyz:
          v = v[0], v[2], v[1]
        self.normals.append(v)
      elif values[0] == 'vt':
        self.texcoords.append(map(float, values[1:3]))
      elif values[0] in ('usemtl', 'usemat'):
        material = values[1]
      elif values[0] == 'mtllib':
        self.mtl = MTL(values[1])
      elif values[0] == 'f':
        face = []
        texcoords = []
        norms = []
        for v in values[1:]:
          w = v.split('/')
          face.append(int(w[0]))
          if len(w) >= 2 and len(w[1]) > 0:
            texcoords.append(int(w[1]))
          else:
            texcoords.append(0)
          if len(w) >= 3 and len(w[2]) > 0:
            norms.append(int(w[2]))
          else:
            norms.append(0)
        self.faces.append((face, norms, texcoords, material))

# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
import sys
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

# IMPORT OBJECT LOADER
from objloader import *

pygame.init()
viewport = (800,600)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
glEnable(GL_LIGHT0)
glEnable(GL_LIGHTING)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded

# LOAD OBJECT AFTER PYGAME INIT
import os
s = ""
l = (sys.argv[1].split("/"))
for fil in l[:-1]:
  s = s + fil +"/"
os.chdir(s)
print(os.getcwd())

obj = OBJ(l[-1], swapyz=True)
