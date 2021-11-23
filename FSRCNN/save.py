

from model import FSRCNN
R = 2
epoch = 24
IS_FSRCNN_S = False

model = FSRCNN(d=32, s=5, m=1, r=R) if IS_FSRCNN_S else FSRCNN(r=R)
model.load_weights("checkpoints/FSRCNN{epoch:03d}.ckpt".format(epoch=epoch) )
model.summary()
