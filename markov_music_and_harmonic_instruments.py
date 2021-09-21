import math as m
import numpy as np
import simpleaudio as sa
import random as r
from scipy import signal
from matplotlib import  pyplot as plt
from scipy.io.wavfile import write


class Note:
    def __init__(self, note, octave, mode_chord, time_duration, symbol):
        self.freq =  self.get_freq(octave, note)
        self.note = idx_to_notes_values[note] + str(octave)
        self.mode_chord = mode_chord
        self.time_duration = time_duration
        self.duration_symbol = symbol

    @staticmethod
    def get_freq(octave, note):
        return 440.0 * m.pow(2, ((octave - 4) + (note - 9) / 12.0))

    def str_mode(self):
        if self.mode_chord == mode_chords_indx["MAYOR"]:
            return "M"
        elif self.mode_chord == mode_chords_indx["MENOR"]:
            return "m"
        elif self.mode_chord == mode_chords_indx["AUMENTADA"]:
            return "aug"
        else:
            return "dis"

    def __str__(self):
        return f"{self.duration_symbol} {self.note}{self.str_mode()} - {round(self.freq,2)}Hz"


# We will use 4 octaves. 3,4,5,6
STARTING_OCTAVE = 3

# INTERVALS
THIRD_MAYOR_RATE = m.pow(2, 4/12)
THIRD_MINOR_RATE = m.pow(2, 3/12)
QUINTA_AUMENTADA_RATE = m.pow(2, 8/12)
QUINTA_JUSTA_RATE = m.pow(2, 7/12)
QUINTA_DISMINUIDA_RATE = m.pow(2, 6/12)


notes_values = {
    "C": 0,   "C#": 1, "D": 2,   "D#": 3,
    "E": 4,   "F": 5,  "F#": 6,  "G": 7,
    "G#": 8,  "A": 9,  "A#": 10, "B": 11
}
idx_to_notes_values = [
    "C",   "C#", "D",   "D#",
    "E",   "F",  "F#",  "G",
    "G#",  "A",  "A#", "B"
]

durations = [0.5, 1, 2, 4]
duration_symbol = ["♪", "♩", "𝅗𝅥", "𝅝"]

"""
Probabilities from:
https://www.researchgate.net/figure/Probabilities-for-the-type-and-duration-of-melody-notes-intervals-between-them-and_fig6_342081971
"""

transition_in_a_octave = [.1, 0,  0.35,  0.1, 0.24,  0.05,  0,  0.1, 0.01, 0.04, 0.01, 0.04]
note_transition = [[transition_in_a_octave[(i-j) % 12]for j in range(12)] for i in range(12)]

octave_transition = [
    [0.6, 0.3, 0.1, 0],
    [0.3, 0.6, 0.3, 0.1],
    [0.1, 0.3, 0.6, 0.3],
    [0, 0.1, 0.3, 0.6],
]

time_transition = [
    [0.5, 0.3, 0.15, 0.05],
    [0.3, 0.5, 0.3, 0.15],
    [0.15, 0.3, 0.5, 0.3],
    [0.05, 0.15, 0.3, 0.5]
]

mode_chords_indx = {"MAYOR": 0,"MENOR": 1, "DISMINUIDA": 2, "AUMENTADA": 3}

mode_transition = [
    [0.63, 0.33, 0.02, 0.02],
    [0.33, 0.63, 0.02, 0.02],
    [0.02, 0.33, 0.33, 0.32],
    [0.33, 0.02, 0.32, 0.63]
]


# Filters
def decay_filter(t,duration):
    return np.exp(-t * 2.0/duration)


def start_ramp_up_filter(t, duration):
    return 1 - np.exp(-t*20.0/duration)


# Markov Transitions
def get_transition(curr_value, transition_matrix):
    result = r.random()
    acc_probability = transition_matrix[curr_value][0]
    next_value = 0
    while acc_probability < result:
        next_value += 1
        acc_probability += transition_matrix[curr_value][next_value]
    return next_value


def get_next_octave(curr_octave):
    return get_transition(curr_octave - STARTING_OCTAVE, octave_transition) + STARTING_OCTAVE


# Harmonic, note creation
def get_harmonic_array(freq, t, lenght=20):
    return np.array([np.sin(2*np.pi * freq * (i+1) * t) for i in range(lenght)])


def get_base_note(freq, t, amps,):
    return np.sum(get_harmonic_array(freq,t,len(amps))* np.array(amps).reshape(len(amps),1), axis=0)


# ------ Instruments ----------
def piano_note(freq, t):
    # ok!
    piano_amps = [0.6, 0.3, 0.04, 0.04, 0.02,
                  0.006, 0.003, 0.01, 0.004, 0.002,
                  0.0006, 0.005]
    return get_base_note(freq, t, piano_amps) * decay_filter(t,t[-1]) * start_ramp_up_filter(t,t[-1]/100)


def brillant_organ_note(freq, t):
    # ok!
    piano_amps = [1, 1.38, 0.874, 0.684]
    return get_base_note(freq, t, piano_amps) * decay_filter(t, t[-1]) *start_ramp_up_filter(t,t[-1]/100)


def organ_note(freq, t):
    # ok! low tempo.
    organ_amps = [0.7, 0.2, 0.03,0.13,
                  0.02,0.01,0.023,0.10,
                  0.02, 0.01, 0.005,0.025,
                  0.010,0.021,0.06,0.05,
                  0.039,0.03,0.021,0.012,
                  0.006, 0.006, 0.003,0.0015]
    return get_base_note(freq/2,t, organ_amps) * decay_filter(t, t[-1]) *start_ramp_up_filter(t,t[-1])


def marimba_note(freq, t):
    # ok!
    marimba_amps = [0.7, 0.2, 0.16,0.13, 0.10,+ 0.07,+ 0.04,+ 0.02,+ 0.02,+ 0.01,+ 0.005, + 0.015,+ 0.010]
    return get_base_note(freq,t,marimba_amps) * decay_filter(t, t[-1]/10) * start_ramp_up_filter(t,t[-1])


def flute_note(freq, t):
    # ok!
    flute_amps = [1, 8, 3.8, 1.8, 0.4, 0.2, 0.04, 0.1, 0.06, 0.08, 0.02,0.015, 0.010]
    return get_base_note(freq,t,flute_amps) * decay_filter(t, t[-1]) * start_ramp_up_filter(t,t[-1]) * (0.7+0.3*np.sin(t))


def oboe_note(freq, t):
    # 2/3
    flute_amps = [1,.98,2.1,0.18, 0.2,0.22, 0.54, 0.30, .20, 0.01, 0.02, 0.05,0.010]
    return get_base_note(2*freq, t, flute_amps) * decay_filter(t, t[-1]) * start_ramp_up_filter(t,t[-1]/10)  * (0.9+0.1*np.sin(4*t))


def horn_note(freq, t):
    # ok!!!
    horn_amps = [1,0.4, 0.23, 0.21, 0.08, 0.06, 0.07, 0.05, 0.04, 0.03, 0.02,0.01, 0.015, 0.005, 0.007, 0.003, 0.001]
    return get_base_note(freq/4, t, horn_amps) * (1 - np.exp(-t * 20.0))* (0.9+0.1*np.sin(2*t))


# ----- intervals -------
def get_third(mode_chord):
    return THIRD_MAYOR_RATE if mode_chord == mode_chords_indx["MAYOR"] \
                                      or mode_chord == mode_chords_indx["AUMENTADA"] else THIRD_MINOR_RATE


def get_quinta(mode_chord):
    return QUINTA_JUSTA_RATE if mode_chord == mode_chords_indx["MAYOR"] \
                                        or mode_chord == mode_chords_indx["MENOR"] else \
        (QUINTA_AUMENTADA_RATE if mode_chords_indx["AUMENTADA"] else QUINTA_DISMINUIDA_RATE)


# ---- chords -----
def create_chord(freq, t,mode_chord, instrument=piano_note):
    atenuante=0.3
    ratio_quinta = get_quinta(mode_chord)
    ratio_third = get_third(mode_chord)
    return (instrument(freq, t) + .3 * instrument(ratio_third * freq, t) \
           + 0.1 * instrument(ratio_quinta * freq, t))* (1-atenuante+atenuante*np.sin(t))


# --- mix instruments, orchesta, etc ---------
def scary_orchesta(freq, t, mode_chord):
    return  10*create_chord(freq, t,mode_chord, instrument=marimba_note)+\
            2*create_chord(freq/2,t,mode_chord,instrument=horn_note)+\
            3*create_chord(freq/get_quinta(mode_chord), t,mode_chord, instrument=horn_note)+\
            create_chord(freq, t,mode_chord, instrument=oboe_note)+\
            2*create_chord(freq/2,t,mode_chord,instrument=horn_note)+\
            3*create_chord(freq/get_quinta(mode_chord), t,mode_chord, instrument=horn_note)


def my_amazing_organ(freq, t, mode_chord):
    chord = 10 * create_chord(freq, t, mode_chord, instrument=piano_note) \
            + 5 * create_chord(freq / get_third(mode_chord), t, mode_chord, instrument=organ_note) \
            + 20 * create_chord(freq / 2, t, mode_chord, instrument=horn_note) \
            + 30 * create_chord(freq / get_quinta(mode_chord), t, mode_chord, instrument=horn_note)

    chord += create_chord(freq, t, mode_chord, instrument=brillant_organ_note) \
             + 3 * create_chord(freq * get_quinta(mode_chord), t, mode_chord, instrument=brillant_organ_note) \
             + 3 * create_chord(2 * freq * get_quinta(mode_chord), t, mode_chord, instrument=brillant_organ_note) \
             + 2 * create_chord(freq * get_third(mode_chord), t, mode_chord, instrument=brillant_organ_note) \
             + 3 * create_chord(2 * freq, t, mode_chord, instrument=brillant_organ_note) \
             + 1 * create_chord(4 * freq * get_quinta(mode_chord), t, mode_chord, instrument=brillant_organ_note) \
             + 1 * create_chord(4 * freq * get_third(mode_chord), t, mode_chord, instrument=brillant_organ_note) \
             + 1.5 * create_chord(4 * freq, t, mode_chord, instrument=brillant_organ_note)
    return chord


def generate_chord(freq, mode_chord, duration, fs):
    # Generate array with duration*sample_rate steps, ranging between 0 and duration
    t = np.linspace(0, duration, duration * fs, False)

    # recommended tempo 100
    # chord = scary_orchesta(freq, t, mode_chord)

    # recommeded tempo 40-50
    chord = my_amazing_organ(freq,t, mode_chord)

    #plt.plot(chord[:])
    #plt.show()

    """
    https://amath.colorado.edu/pub/matlab/music/MathMusic.pdf
    """
    return chord


def play_song(notes, name="random" ,overlap_ratio=0.2):
    # Function got from: https://realpython.com/playing-and-recording-sound-python/
    print("Preparing song . . .")
    fs = 44100  # 44100 samples per second
    song = np.array([])

    for i in range(len(notes)):
        chord = generate_chord(notes[i].freq,notes[i].mode_chord, notes[i].time_duration*(1+overlap_ratio), fs)
        if overlap_ratio == 0:
            song = np.append(song, chord)
        else:
            if i == 0:
                song = np.append(song, chord)
            else:
                overlap = int(overlap_ratio*len(chord))
                song[len(song) - overlap:] += chord[:overlap]
                song = np.append(song, chord[overlap:])


    # Ensure that highest value is in 16-bit range
    audio = song * (2 ** 15 - 1) / np.max(np.abs(song))
    # Convert to 16-bit data
    audio = audio.astype(np.int16)

    if name == "random":
        name += str(int(r.random()*100))
    write(f"{name}.wav", fs, audio)
    while True:
        print("Ready: Play enter.")
        x = input()
        if x:
            # Start playback
            play_obj = sa.play_buffer(audio, 1, 2, fs)

            # Wait for playback to finish before exiting
            play_obj.wait_done()

    return None


def play(duracion_cancion_segundos=120, tempo=60, first_note="B", first_octave=3):
    unit_of_time = (60/tempo)  # in seconds
    curr_note_duration_index = 1
    curr_note = notes_values[first_note]
    curr_octave = first_octave
    curr_mode = mode_chords_indx["MAYOR"]
    notes = []

    while duracion_cancion_segundos > 0:
        curr_note = get_transition(curr_note, note_transition)
        curr_octave = get_next_octave(curr_octave)
        mode_chord = get_transition(curr_mode, mode_transition)
        curr_duration_index = get_transition(curr_note_duration_index, time_transition)
        time_duration = durations[curr_duration_index] * unit_of_time

        duracion_cancion_segundos -= time_duration
        notes.append(Note(curr_note, curr_octave, mode_chord, time_duration, duration_symbol[curr_duration_index]))

    for n in notes:
        print(n)
    play_song(notes, name="volare")


# Songs created by me.
def volare_song():
    tempo = 100
    notes = ["D","G","F#","A", "G", "D", "B", "E", "A", "B", "B", "A", "C", "B", "G", "G",
             "F#", "D", "G","F#","D", "B", "D", "A", "C", "B", "A", "G", "F#"]
    durations = [3,   6, 1.5, 1.5,  4,   1,   1,   5,   1,    1,    8, 1,   1,   1,   1,   1,
                 1,  1,   1,  1,   1,   1,   6, 1,    1,   1,   1,  1,    4]
    octaves = [3,   3,  3,   3,   3,   3,   2,   3,   3,    3,    3, 3,    4,
               3,   3,   4,  4,   4,  4,   4,   4,  3,   4, 3,   4,   3,    3,   3,   3]
    mode_chord = 0

    song = []

    for i in range(len(notes)):
        song.append(Note(notes_values[notes[i]], octaves[i], mode_chord, (60/tempo) * durations[i], ""))

    play_song(song, 0)


def zarathustra_song():
    tempo = 100

    notes = [
        "C", "G", "C", "E", "D#",
        "G", "C", "G", "C", "G", "C", "G", "C",

        "C", "G", "C", "E", "D#",
        "G", "C", "G", "C", "G", "C", "G", "C",

        "C", "G", "C", "D#", "E",
        "E", "F", "G", "A", "C",
                 ]

    durations = [
        4, 4, 3.5, .5, 4,
        0.5, 0.5,
        0.5, 0.5,
        0.5, 0.5,
        0.5, 0.5,

        4, 4, 3.5, .5, 4,
        0.5, 0.5,
        0.5, 0.5,
        0.5, 0.5,
        0.5, 0.5,

        4, 4, 3.5, 1, 3.5,
        2, 1, 3, 4, 8
    ]

    octaves = [
        2, 2, 3, 3, 3,
        2, 2, 2, 2,
        2, 2, 2, 2,

        2, 2, 3, 3, 3,
        2, 2, 2, 2,
        2, 2, 2, 2,

        2, 2, 3, 3, 3,
        3, 3, 3, 3, 4]

    mode_chord = 0

    song =[]

    for i in range(len(notes)):
        song.append(Note(notes_values[notes[i]], octaves[i], mode_chord, (60/tempo) * durations[i], ""))

    play_song(song,name="zarathustra", overlap_ratio=0)


if __name__ == "__main__":
    # play()
    # volare_song()
    zarathustra_song()
