import os

def make_noise(duration,freq):
    '''Make noise after finishing executing a code'''
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

def finish():
    make_noise(0.2,320)
    make_noise(0.2,350)
    make_noise(0.2,400)

finish()