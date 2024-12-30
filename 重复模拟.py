import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import commpy as cpy

# 定义制调制方式
def Modulation(bits):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        symbol = PSK4.modulate(bits)
        return symbol
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        symbol = QAM64.modulate(bits)
        return symbol
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        symbol = QAM16.modulate(bits)
        return symbol
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        symbol = PSK8.modulate(bits)
        return symbol
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        symbol = BPSK.modulate(bits)
        return symbol

# 定义解调方式
def DeModulation(symbol):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        bits = PSK4.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        bits = QAM64.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        bits = QAM16.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        bits = PSK8.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        bits = BPSK.demodulate(symbol, demod_type='hard')
        return bits

# 信道函数
def add_awgn(x_s, snrDB):
    data_pwr = np.mean(abs(x_s**2))
    noise_pwr = data_pwr/(10**(snrDB/10))
    noise = 1/np.sqrt(2) * (np.random.randn(len(x_s)) + 1j *
                            np.random.randn(len(x_s))) * np.sqrt(noise_pwr)
    return x_s + noise, noise_pwr

def channel(in_signal, channelResponse):
    out_signal = np.convolve(in_signal, channelResponse)
    return out_signal

def random_complex_channel(N, amplitude_range=(0.1, 1.0), phase_range=(0, 2*np.pi)):
    amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], N)
    phases = np.random.uniform(phase_range[0], phase_range[1], N)
    channelResponse = amplitudes * (np.cos(phases) + 1j * np.sin(phases))
    return channelResponse

# 插入导频和数据，生成OFDM符号
def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex)  # 子载波位置
    symbol[pilotCarriers] = pilotValue  # 在导频位置插入导频
    symbol[dataCarriers] = QAM_payload  # 在数据位置插入数据
    return symbol

# 快速傅里叶逆变换
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

# 添加循环前缀
def addCP(OFDM_time):
    cp = OFDM_time[-CP:]  # CP=16
    return np.hstack([cp, OFDM_time])

# 接收端，去除循环前缀
def removeCP(signal):
    return signal[CP:(CP+K)]

# 快速傅里叶变换
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

# 信道估计
def channelEstimate(Hest_at_pilots):
    Hest_abs = interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    return Hest

# 均衡
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]

# 主循环，N次迭代
N = 1000  # Number of iterations for averaging BER
ber_values_avg = []

pilotValue = 3+3j  # 导频格式
K = 64  # OFDM子载波数量
P_values = range(1, K//2 + 1, 1)  # 可以调整导频数
allCarriers = np.arange(K)  # 子载波编号 ([0, 1, ... K-1])

channelResponse = random_complex_channel(3)  # 信道响应

CP = K // 4  # 25%的循环前缀长度
Modulation_type = 'QAM16'  # 调制方式，可选BPSK、QPSK、8PSK、QAM16、QAM64
m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
mu = m_map[Modulation_type]  # 调制后一个符号携带比特数

for P in P_values:
    pilot_Interval = K // P  # 每间隔P个子载波一个导频
    pilotCarrier = allCarriers[::pilot_Interval]  # 每间隔P个子载波一个导频
    pilotCarriers = np.hstack([pilotCarrier, np.array([allCarriers[-1]])])  # 导频载波
    P = P + 1  # 导频的数量加1
    dataCarriers = np.delete(allCarriers, pilotCarriers)  # 携带信息量的载波，去除了导频载波
    payloadBits_per_OFDM = len(dataCarriers) * mu  # 每个 OFDM 符号的有效比特数

    # 计算 N 次迭代中的误比特率平均值
    ber_values = []
    for _ in range(N):
        # 产生比特流
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))  # 生成随机比特流，0，1等概

        # 比特信号调制
        QAM_s = Modulation(bits)
        OFDM_data = OFDM_symbol(QAM_s)

        # 快速逆傅里叶变换
        OFDM_time = IDFT(OFDM_data)

        # 添加循环前缀
        OFDM_withCP = addCP(OFDM_time)  # 在OFDM符号前加16个前缀

        # 经过信道
        OFDM_TX = OFDM_withCP
        OFDM_RX = channel(OFDM_TX, channelResponse)

        # 去除循环前缀
        OFDM_RX_noCP = OFDM_RX[CP:(CP + K)]

        # 快速傅里叶变换
        OFDM_demod = DFT(OFDM_RX_noCP)  # 恢复的QAM16信号

        # 信道估计
        pilots = OFDM_demod[pilotCarriers]  # 取导频处的数据
        Hest_at_pilots = pilots / pilotValue  # LS信道估计s
        Hest = channelEstimate(Hest_at_pilots)

        # 均衡
        equalized_Hest = equalize(OFDM_demod, Hest)

        # 获取数据位置的数据
        QAM_est = get_payload(equalized_Hest)

        # 反映射，解调
        bits_est = DeModulation(QAM_est)

        # 计算误比特率（BER）
        ber = np.sum(abs(bits - bits_est)) / len(bits)
        ber_values.append(ber)

    # 计算 N 次迭代后的平均 BER
    avg_ber = np.mean(ber_values)
    ber_values_avg.append(avg_ber)

# 绘制不同导频数对误比特率影响的图像
plt.plot(P_values, ber_values_avg, marker='o')
plt.xlabel('Number of Pilots')
plt.ylabel('Average Bit Error Rate (BER)')
plt.title('Average BER vs Number of Pilots (over N iterations)')
plt.grid(True)
for i, value in enumerate(ber_values_avg):
    plt.text(P_values[i], value, f'{value:.4f}', fontsize=5, ha='center', va='bottom')

plt.show()
