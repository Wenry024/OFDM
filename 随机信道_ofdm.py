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
# 定义信道
def add_awgn(x_s, snrDB):
    data_pwr = np.mean(abs(x_s**2))
    noise_pwr = data_pwr/(10**(snrDB/10))
    noise = 1/np.sqrt(2) * (np.random.randn(len(x_s)) + 1j *
                            np.random.randn(len(x_s))) * np.sqrt(noise_pwr)
    return x_s + noise, noise_pwr
def channel(in_signal,channelResponse):
    out_signal = np.convolve(in_signal, channelResponse)

    return out_signal


# def channel(in_signal, noise_std=0.01, fading_param=None):
#     """
#     模拟一个平坦衰落信道。
#
#     参数:
#     input_signal (numpy.array): 输入信号，一个复数数组。
#     noise_std (float): 噪声的标准差，默认为0.01。
#     fading_param (tuple): 包含直射分量和非直视分量的元组，默认为None。
#
#     返回:
#     output_signal (numpy.array): 经过平坦衰落信道的输出信号。
#     """
#
#     # 创建一个平坦衰落信道对象
#     siso_channel = SISOFlatChannel(fading_param=fading_param, noise_std=noise_std)
#
#     # 通过信道传递信号
#     output_signal = siso_channel(input_signal)
#
#     return output_signal
#
#
# import numpy as np

def random_complex_channel(N, amplitude_range=(0.1, 1.0), phase_range=(0, 2*np.pi)):
    """
    生成一个具有随机复数元素的信道冲激响应。

    参数:
    N (int): 冲激响应的长度。
    amplitude_range (tuple): 复数幅度的范围，默认为(0.1, 1.0)。
    phase_range (tuple): 复数相位的范围，默认为(0, 2*pi)。

    返回:
    channel_response (numpy.array): 随机复数信道冲激响应。
    """
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
    cp = OFDM_time[-CP:] #CP=16
    return np.hstack([cp, OFDM_time])
# 接收端，去除循环前缀
def removeCP(signal):
    return signal[CP:(CP+K)]

# 快速傅里叶变换
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)
# 信道估计
def channelEstimate(Hest_at_pilots):

    # 在导频载波之间进行插值以获得估计，然后利用插值估计得到数据下标处的信道响应
    Hest_abs = interpolate.interp1d(pilotCarriers, abs(
        Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = interpolate.interp1d(pilotCarriers, np.angle(
        Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    return Hest
# 均衡
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def get_payload(equalized):
    return equalized[dataCarriers]



K = 64#OFDM子载波数量
P = 22 # 导频数
P_Interval=K//P# 导频间隔
pilotValue = 3+3j  # 导频格式
allCarriers = np.arange(K)  # 子载波编号 ([0, 1, ... K-1])
pilotCarrier = allCarriers[::P_Interval]  # 每间隔P个子载波一个导频
# 为了方便信道估计，将最后一个子载波也作为导频
pilotCarriers = np.hstack([pilotCarrier, np.array([allCarriers[-1]])])
P = P+1 # 导频的数量也需要加1

CP = K//4  #25%的循环前缀长度
Modulation_type = 'QAM16' #调制方式，可选BPSK、QPSK、8PSK、QAM16、QAM64
channel_type ='awgn' # 信道类型，可选awgn
SNRdb = 25  # 接收端的信噪比（dB）
m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
mu = m_map[Modulation_type]#调制后一个符号携带比特数
dataCarriers = np.delete(allCarriers, pilotCarriers)#携带信息量的载波，去除了导频载波
payloadBits_per_OFDM = len(dataCarriers)*mu  # 每个 OFDM 符号的有效比特数

# 产生比特流
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))#生成随机比特流，0，1等概
# 进行卷积编码
# channelResponse = np.array([1, 0, 0.3+0.3j])  # 随意仿真信道冲激响应
channelResponse=random_complex_channel(3)
# 比特信号调制
QAM_s = Modulation(bits)
OFDM_data = OFDM_symbol(QAM_s)
# 快速逆傅里叶变换
OFDM_time = IDFT(OFDM_data)
# 添加循环前缀
OFDM_withCP = addCP(OFDM_time)#在OFDM符号前加16个前缀

# 经过信道
OFDM_TX = OFDM_withCP
OFDM_RX = channel(OFDM_TX, channelResponse)




# 去除循环前缀
# OFDM_RX_noCP = removeCP(OFDM_RX)
OFDM_RX_noCP=OFDM_RX[CP:(CP+K)]
# 快速傅里叶变换
OFDM_demod = DFT(OFDM_RX_noCP)#恢复的QAM16信号

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

# print(bits_est)
print("误比特率BER： ", np.sum(abs(bits-bits_est ))/len(bits))


#可视化调制后的星座图
plt.plot(QAM_s.real, QAM_s.imag, 'bo')


plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary Part')
plt.title("Received constellation")
plt.savefig('map.png')
plt.show()

# 可视化数据和导频的插入方式
dataCarriers = np.delete(allCarriers, pilotCarriers)
plt.figure(figsize=(8, 0.8))
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
plt.legend(fontsize=10, ncol=2)
plt.xlim((-1, K))
plt.ylim((-0.1, 0.3))
plt.xlabel('Carrier index')
plt.yticks([])
plt.grid(True)
plt.savefig('carrier.png')
plt.show()

# 可视化信道冲击响应，仿真信道
H_exact = np.fft.fft(channelResponse, K)
plt.plot(allCarriers, abs(H_exact))
plt.xlabel('Subcarrier index')
plt.ylabel('$|H(f)|$')
plt.grid(True)
plt.xlim(0, K-1)
plt.show()

OFDM_RX = channel(OFDM_TX,  channelResponse)
plt.figure(figsize=(8,2))
plt.plot(abs(OFDM_TX), label='TX signal')
plt.plot(abs(OFDM_RX), label='RX signal')
plt.legend(fontsize=10)
plt.xlabel('Time'); plt.ylabel('$|x(t)|$')
plt.grid(True)
plt.show()

# 信道估计
plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
plt.scatter(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
plt.grid(True)
plt.xlabel('Carrier index')
plt.ylabel('$|H(f)|$')
plt.legend(fontsize=10)
plt.ylim(0, 2)
plt.savefig('信道响应估计.png')
plt.show()

# 可视化均衡后的星座图
plt.plot(QAM_est.real, QAM_est.imag, 'bo')
plt.plot(QAM_s.real, QAM_s.imag, 'ro')

plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary Part')
plt.title("Received constellation")
plt.savefig('map.png')
plt.show()

# 可视化发送端QAM16调制后的星座图
plt.figure(figsize=(6, 6))
plt.plot(QAM_s.real, QAM_s.imag, 'bo')  # 发送端的QAM16调制符号
plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary Part')
plt.title("Transmitted QAM16 Constellation")
plt.savefig('transmitted_constellation.png')
plt.show()

# 可视化接收端的星座图（均衡前）
plt.figure(figsize=(6, 6))
plt.plot(OFDM_demod.real, OFDM_demod.imag, 'ro')  # 接收端未均衡的符号
plt.plot(QAM_s.real, QAM_s.imag, 'bo', label='Transmitted symbols')
plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary Part')
plt.title("Received QAM16 Constellation Before Equalization")
plt.savefig('received_before_equalization.png')
plt.show()

# 可视化均衡后的星座图
plt.figure(figsize=(6, 6))
plt.plot(QAM_est.real, QAM_est.imag, 'ro', label='Equalized symbols')  # 接收端均衡后的符号
plt.plot(QAM_s.real, QAM_s.imag, 'bo', label='Transmitted symbols')  # 原始发送符号
plt.legend()
plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary Part')
plt.title("Equalized QAM16 Constellation")
plt.savefig('equalized_constellation.png')
plt.show()
