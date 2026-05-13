import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Параметры фильтра
fs = 1000.0  # Частота дискретизации (Гц)
fc = 100.0  # Частота среза (Гц)
order = 60  # Порядок фильтра
# Длина окна
M = order + 1

# Нормированная частота среза (от 0 до 1, где 1 соответствует fs/2)
nyq = fs / 2
Wn = fc / nyq

print("=" * 60)
print(f"Параметры фильтра:")
print(f"  Частота дискретизации: {fs} Гц")
print(f"  Частота Найквиста: {nyq} Гц")
print(f"  Нормированная частота среза: {Wn:.3f} (1 = {nyq} Гц)")
print("=" * 60)

# 2. Расчёт коэффициентов КИХ-фильтра методом окна Хэмминга
n = np.arange(M)
# Защита от деления на ноль для центрального отсчёта
with np.errstate(divide='ignore', invalid='ignore'):
    h_ideal = np.sin(np.pi * Wn * (n - (M - 1) / 2)) / (np.pi * (n - (M - 1) / 2))

center_index = (M - 1) // 2
h_ideal[center_index] = Wn

# Окно Хэмминга
window = np.hamming(M)

# Итоговая импульсная характеристика
h = h_ideal * window

# 3. Расчёт и визуализация импульсной характеристики
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.stem(n, h, basefmt=" ")
plt.title('Импульсная характеристика h[n] (окно Хэмминга)')
plt.xlabel('Отсчёт n')
plt.ylabel('Амплитуда')
plt.grid(True)

# 4. Расчёт АЧХ и ФЧХ
# Используем частотную характеристику в нормированных частотах
w_norm, H = signal.freqz(h, worN=8192)  # w_norm в радианах/выборка
# Преобразуем в нормированную частоту от 0 до 1 (где 1 = fs/2)
f_norm = w_norm / np.pi

# АЧХ
H_dB = 20 * np.log10(np.abs(H))

plt.subplot(2, 2, 2)
plt.plot(f_norm, H_dB, 'b', linewidth=1.5)
plt.axvline(Wn, color='r', linestyle='--', linewidth=1.5, label=f'Норм. частота среза = {Wn:.3f}')

plt.ylim(-150, 5)
plt.xlim(0, 1)
plt.title('АЧХ фильтра (окно Хэмминга)')
plt.xlabel('Нормированная частота')
plt.ylabel('Амплитуда (дБ)')
plt.grid(True)
plt.legend()

# ФЧХ
plt.subplot(2, 2, 3)
plt.plot(f_norm, np.unwrap(np.angle(H)), 'g', linewidth=1.5)
plt.title('ФЧХ фильтра')
plt.xlabel('Нормированная частота')
plt.ylabel('Фаза (рад)')
plt.grid(True)

# 5. Исследование влияния порядка фильтра
orders_to_test = [14, 30, 50, 100]
plt.subplot(2, 2, 4)

for ord_val in orders_to_test:
    M_test = ord_val + 1
    n_test = np.arange(M_test)
    # Идеальная ИХ
    h_ideal_test = np.sin(np.pi * Wn * (n_test - (M_test - 1) / 2)) / (np.pi * (n_test - (M_test - 1) / 2))
    h_ideal_test[(M_test - 1) // 2] = Wn
    # Окно Хэмминга
    window_test = np.hamming(M_test)
    h_test = h_ideal_test * window_test
    # АЧХ в нормированных частотах
    w_test_norm, H_test = signal.freqz(h_test, worN=8192)
    f_test_norm = w_test_norm / np.pi
    H_test_dB = 20 * np.log10(np.abs(H_test) + 1e-12)

    plt.plot(f_test_norm, H_test_dB, linewidth=1.5, label=f'Порядок = {ord_val}')

# Добавляем линию частоты среза
plt.axvline(Wn, color='r', linestyle='--', linewidth=1.5, label=f'Норм. частота среза = {Wn:.3f}')
plt.ylim(-150, 5)
plt.xlim(0, 1)
plt.title('Влияние порядка фильтра')
plt.xlabel('Нормированная частота')
plt.ylabel('Амплитуда (дБ)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 6. Фильтрация тестового сигнала (синус + шум)
# Генерация тестового сигнала: полезный сигнал на 50 Гц, шум — высокочастотный (250 Гц)
t = np.arange(0, 1.0, 1 / fs)  # 1 секунда
f_signal = 50.0
signal_clean = np.sin(2 * np.pi * f_signal * t)
noise_amplitude = 0.5
noise_freq = 250.0
noise = noise_amplitude * np.sin(2 * np.pi * noise_freq * t)
# Добавление случайного шума
noise += 0.2 * np.random.randn(len(t))
signal_noisy = signal_clean + noise

# Применяем фильтр
filtered_signal = signal.lfilter(h, 1.0, signal_noisy)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t, signal_clean, 'g', linewidth=1.5, label='Исходный сигнал (50 Гц)')
plt.xlim(0, 0.2)
plt.title('Исходный чистый сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, signal_noisy, 'r', alpha=0.7, label='Зашумлённый сигнал')
plt.xlim(0, 0.2)
plt.title('Тестовый сигнал с высокочастотной помехой (250 Гц + шум)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, 'b', linewidth=1.5, label='Выход фильтра')
plt.plot(t, signal_clean, 'g--', linewidth=1, alpha=0.7, label='Ожидаемый сигнал')
plt.xlim(0, 0.2)
plt.title('Результат фильтрации (сравнение с теоретическим)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 8. Сравнение пульсаций + длина переходной полосы
def get_filter_metrics(ord, Wn, nyq, fs):
    """Возвращает: ширина перехода, пульсации в ПП, пульсации в ПЗ"""
    M_loc = ord + 1
    n_loc = np.arange(M_loc)

    with np.errstate(divide='ignore', invalid='ignore'):
        h_ideal_loc = np.sin(np.pi * Wn * (n_loc - (M_loc - 1) / 2)) / (np.pi * (n_loc - (M_loc - 1) / 2))
    h_ideal_loc[(M_loc - 1) // 2] = Wn
    h_loc = h_ideal_loc * np.hamming(M_loc)

    # АЧХ
    w_loc, H_loc = signal.freqz(h_loc, worN=8192)
    f_norm_loc = w_loc / np.pi
    H_dB_loc = 20 * np.log10(np.abs(H_loc) + 1e-12)
    H_abs_loc = np.abs(H_loc)

    idx_3db = np.where(H_abs_loc <= 0.707)[0]
    f_3db_norm = f_norm_loc[idx_3db[0]] if len(idx_3db) > 0 else Wn

    idx_40db = np.where(H_abs_loc <= 0.01)[0]
    f_40db_norm = f_norm_loc[idx_40db[0]] if len(idx_40db) > 0 else 1.0

    trans_width_hz = (f_40db_norm - f_3db_norm) * nyq


    pb = H_dB_loc[f_norm_loc <= Wn]
    ripple_pp = np.max(pb) - np.min(pb) if len(pb) > 0 else 0

    sb_start = np.where((f_norm_loc >= Wn) & (H_dB_loc <= -20))[0]
    if len(sb_start) > 0:
        sb = H_dB_loc[sb_start[0]:]
        ripple_sb = np.max(sb) - np.min(sb) if len(sb) > 0 else 0
    else:
        ripple_sb = 0

    return trans_width_hz, ripple_pp, ripple_sb

print("\n" + "=" * 75)
print(f"{'Порядок':>8} | {'Переход (Гц)':>13} | {'ПП: Δ (дБ)':>10} | {'ПЗ: Δ (дБ)':>12}")
print("-" * 75)

for ord_val in sorted(orders_to_test):
    trans_w, r_pp, r_sb = get_filter_metrics(ord_val, Wn, nyq, fs)
    print(f"{ord_val:8d} | {trans_w:13.2f} | {r_pp:10.4f} | {r_sb:12.2f}")

print("=" * 75)