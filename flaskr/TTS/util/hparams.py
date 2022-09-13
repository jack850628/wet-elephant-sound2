from .symbols import symbols

max_char = 105
sample_rate = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
preemphasis = 0.97
ref_db = 20
max_db = 100
mel_dim = 80
max_length = 780
reduction = 5
embedding_dim = 128#256
decoder_dim = 128#256
symbol_length = len(symbols)#70
batch_size = 32#16
checkpoint_step = 500
max_iter = 200

reference_filters=[32, 32, 64, 64, 128, 128]
reference_depth=128
reference_conv_kernel_size=(3,3)
reference_conv_strides=(2,2)

num_gst=10
num_heads=4
style_embed_depth=256
style_att_dim=128