
from image_test_space import DisplayImage


screen_size = 16
img_size = 512
img_mod = DisplayImage(img_size=img_size, screen_size=screen_size)

assert not img_mod.in_corner(256,256,256+16,256+16)
assert img_mod.in_corner(0,0,16,16)
assert img_mod.in_corner(0,512-16,16,512)
assert not img_mod.in_corner(0,256,16,256+16)
