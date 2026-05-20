import pandas as pd
import numpy as np
import joblib
from scapy.all import sniff, IP, TCP, UDP, ICMP
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. EĞİTİLMİŞ MODELLERİ YÜKLE
# =====================================================
print("[*] Modeller yükleniyor...")

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
threshold_data = joblib.load("threshold.pkl")

cluster_map = threshold_data["map"]
threshold = threshold_data["threshold"]

print(f"[+] Eşik Değeri: {threshold:.4f}")

# =====================================================
# 2. CANLI TRAFİK DATAFRAME
# =====================================================
columns = [
    "src_ip", "dst_ip",
    "src_bytes", "protocol_type", "flag",
    "distance", "prediction"
]

traffic_df = pd.DataFrame(columns=columns)

# =====================================================
# 3. FEATURE ÇIKARMA FONKSİYONU
# =====================================================
def extract_features(packet):
    if not packet.haslayer(IP):
        return None

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    src_bytes = len(packet)

    # --- Protocol ---
    if packet.haslayer(TCP):
        protocol = "tcp"
        flag = packet[TCP].flags
    elif packet.haslayer(UDP):
        protocol = "udp"
        flag = "NONE"
    elif packet.haslayer(ICMP):
        protocol = "icmp"
        flag = "NONE"
    else:
        return None

    return src_ip, dst_ip, src_bytes, protocol, flag

# =====================================================
# 4. IDS KARAR FONKSİYONU
# =====================================================
def detect_anomaly(packet):
    global traffic_df

    data = extract_features(packet)
    if data is None:
        return

    src_ip, dst_ip, src_bytes, protocol, flag = data

    # --- DataFrame satırı ---
    row = pd.DataFrame([{
        "src_bytes": np.log1p(src_bytes),
        "protocol_type": protocol,
        "flag": str(flag)
    }])

    # --- Encoding ---
    try:
        row["protocol_type"] = encoders["protocol_type"].transform(row["protocol_type"])
        row["flag"] = encoders["flag"].transform(row["flag"])
    except:
        return

    # --- Scaling ---
    X_scaled = scaler.transform(row)

    # --- Mesafe Hesabı ---
    distances = kmeans.transform(X_scaled)
    min_distance = np.min(distances)

    # --- Anomali Kararı ---
    if min_distance > threshold:
        prediction = "ANOMALİ"
        print(f"[!] ANOMALİ TESPİT EDİLDİ | {src_ip} → {dst_ip} | Mesafe: {min_distance:.4f}")
    else:
        prediction = "NORMAL"

    # --- Loglama ---
    traffic_df.loc[len(traffic_df)] = [
        src_ip, dst_ip,
        src_bytes, protocol, flag,
        min_distance, prediction
    ]

# =====================================================
# 5. CANLI AĞI DİNLE
# =====================================================
print("\n[*] Canlı ağ trafiği dinleniyor... (CTRL+C ile durdur)")

sniff(prn=detect_anomaly, store=False)
