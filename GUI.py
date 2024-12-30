import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import commpy as cpy


def run_simulation():
    global P, pilotCarriers, dataCarriers, mu, Modulation_type, pilotValue

    try:
        P = int(p_value.get())
        Modulation_type = modulation_type.get()

        if P < 1 or P >= K // 2:
            raise ValueError("P must be in the range [1, K//2].")

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))
        return

    pilotCarrier = allCarriers[::K // P]  # 每间隔P个子载波一个导频
    pilotCarriers = np.hstack([pilotCarrier, np.array([allCarriers[-1]])])
    dataCarriers = np.delete(allCarriers, pilotCarriers)

    P = len(pilotCarriers)
    payloadBits_per_OFDM = len(dataCarriers) * mu

    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
    channelResponse = random_complex_channel(3)
    QAM_s = Modulation(bits)
    OFDM_data = OFDM_symbol(QAM_s)
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_RX = channel(OFDM_withCP, channelResponse)
    OFDM_RX_noCP = removeCP(OFDM_RX)
    OFDM_demod = DFT(OFDM_RX_noCP)
    pilots = OFDM_demod[pilotCarriers]
    Hest_at_pilots = pilots / pilotValue
    Hest = channelEstimate(Hest_at_pilots)
    equalized_Hest = equalize(OFDM_demod, Hest)
    QAM_est = get_payload(equalized_Hest)
    bits_est = DeModulation(QAM_est)
    ber = np.sum(abs(bits - bits_est)) / len(bits)

    ber_label.config(text=f"误比特率 (BER): {ber:.6f}")

    # Plotting
    plt.figure()
    plt.plot(QAM_s.real, QAM_s.imag, 'bo', label="Transmitted")
    plt.title("Transmitted Constellation")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid()
    plt.savefig("transmitted_constellation.png")

    plt.figure()
    plt.plot(QAM_est.real, QAM_est.imag, 'ro', label="Equalized")
    plt.title("Equalized Constellation")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid()
    plt.savefig("equalized_constellation.png")

    plt.figure()
    plt.plot(allCarriers, abs(Hest), label="Estimated Channel")
    plt.title("Channel Estimate")
    plt.xlabel("Subcarrier Index")
    plt.ylabel("|H(f)|")
    plt.grid()
    plt.legend()
    plt.savefig("channel_estimate.png")

    # Show plots
    plt.show()


# GUI setup
root = tk.Tk()
root.title("OFDM Simulation")
root.geometry("400x300")

frame = tk.Frame(root)
frame.pack(pady=10)

# Modulation Type
modulation_label = tk.Label(frame, text="调制方式:")
modulation_label.grid(row=0, column=0, padx=10, pady=5)
modulation_type = ttk.Combobox(frame, values=["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"], state="readonly")
modulation_type.set("QAM16")
modulation_type.grid(row=0, column=1, padx=10, pady=5)

# P Value
p_label = tk.Label(frame, text="P值 (1 - K//2):")
p_label.grid(row=1, column=0, padx=10, pady=5)
p_value = tk.Entry(frame)
p_value.insert(0, "22")
p_value.grid(row=1, column=1, padx=10, pady=5)

# Run button
run_button = tk.Button(root, text="开始仿真", command=run_simulation)
run_button.pack(pady=10)

# BER display
ber_label = tk.Label(root, text="误比特率 (BER): N/A")
ber_label.pack(pady=5)

# Constants and helper functions
K = 64
pilotValue = 3 + 3j
allCarriers = np.arange(K)
m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
mu = m_map["QAM16"]


def random_complex_channel(N, amplitude_range=(0.1, 1.0), phase_range=(0, 2 * np.pi)):
    amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], N)
    phases = np.random.uniform(phase_range[0], phase_range[1], N)
    return amplitudes * (np.cos(phases) + 1j * np.sin(phases))


def Modulation(bits):
    modem = cpy.PSKModem(4) if Modulation_type == "QPSK" else cpy.QAMModem(16)
    return modem.modulate(bits)


def DeModulation(symbol):
    modem = cpy.PSKModem(4) if Modulation_type == "QPSK" else cpy.QAMModem(16)
    return modem.demodulate(symbol, demod_type='hard')


def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex)
    symbol[pilotCarriers] = pilotValue
    symbol[dataCarriers] = QAM_payload
    return symbol


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time):
    cp = OFDM_time[-K // 4:]
    return np.hstack([cp, OFDM_time])


def removeCP(signal):
    return signal[K // 4:]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def channelEstimate(Hest_at_pilots):
    Hest_abs = interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    return Hest_abs * np.exp(1j * Hest_phase)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def get_payload(equalized):
    return equalized[dataCarriers]


def channel(in_signal, channelResponse):
    return np.convolve(in_signal, channelResponse)[:len(in_signal)]


root.mainloop()
