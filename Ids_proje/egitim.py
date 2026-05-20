import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- 1. VERİ YÜKLEME ---
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty_level'
]

train_path = "KDDTrain+.txt"
test_path = "KDDTest+.txt"

print("[*] Veriler yükleniyor...")
try:
    df_train = pd.read_csv(train_path, header=None, names=columns, quoting=3)
    df_test = pd.read_csv(test_path, header=None, names=columns, quoting=3)
except FileNotFoundError:
    print("Hata: Dosyalar bulunamadı.")
    exit()

# --- 2. VERİ TEMİZLEME ---
def clean_data(df):
    df = df.copy()
    df['targets'] = df['class'].apply(lambda x: 0 if 'normal' in str(x).lower() else 1)
    df['src_bytes'] = pd.to_numeric(df['src_bytes'].astype(str).str.replace('"', '').str.strip(), errors='coerce').fillna(0)
    df['src_bytes'] = np.log1p(df['src_bytes'])
    return df

df_train = clean_data(df_train)
df_test = clean_data(df_test)

# --- 3. ENCODING ---
encoders = {}
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    combined = pd.concat([df_train[col], df_test[col]], axis=0).astype(str)
    le.fit(combined)
    df_train[col] = le.transform(df_train[col].astype(str))
    df_test[col] = le.transform(df_test[col].astype(str))
    encoders[col] = le

# --- 4. ÖZELLİK SEÇİMİ ---
final_features = ['src_bytes', 'protocol_type', 'flag']
print(f"[*] Eğitimde Kullanılan Özellikler: {final_features}")

X_train = df_train[final_features]
y_train = df_train['targets']
X_test = df_test[final_features]
y_test = df_test['targets']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PCA EKLENDİ ---
from sklearn.decomposition import PCA
print("\n[*] PCA uygulanıyor...")
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"[+] PCA Sonrası Boyut: {X_train_pca.shape}")
print(f"[+] Açıklanan Varyans Oranları: {pca.explained_variance_ratio_}\n")

# --- 5. K-MEANS ---
print(f"[*] K-Means (k=5) eğitiliyor...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

# ============= EK 2: Elbow Method =======================
import matplotlib.pyplot as plt

print("\n[*] Elbow Method Hesaplanıyor...")
distortions = []
K = range(2, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train_scaled)
    distortions.append(km.inertia_)

plt.figure()
plt.plot(K, distortions, marker='o')
plt.xlabel("K Değeri")
plt.ylabel("Distortion (Inertia)")
plt.title("Elbow Method - K Seçimi")
plt.grid()
plt.show()

# ============= EK 3: PCA Görselleştirme =================
print("\n[*] PCA Sonuçları Görselleştiriliyor...")
plt.figure()
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=y_train, cmap='coolwarm', s=3)
plt.title("PCA - Normal vs Attack Dağılımı")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# --- 6. EŞİK ---
print("[*] Eşik hesaplanıyor...")
cluster_map = {}
train_clusters = kmeans.labels_
distances = kmeans.transform(X_train_scaled)
min_dists = np.min(distances, axis=1)

all_normal_distances = []

for i in range(5):
    indices = np.where(train_clusters == i)
    if len(indices[0]) > 0:
        cluster_type = y_train.iloc[indices].mode()[0]
        cluster_map[i] = cluster_type
        if cluster_type == 0:
            all_normal_distances.extend(min_dists[indices])
    else:
        cluster_map[i] = 0

TARGET_PERCENTILE = 99.2

if len(all_normal_distances) > 0:
    main_threshold = np.percentile(all_normal_distances, TARGET_PERCENTILE)
else:
    main_threshold = 2.5

print("\n" + "="*50)
print(f"[*] KULLANILAN PERCENTILE: %{TARGET_PERCENTILE}")
print(f"[*] HESAPLANAN OTOMATİK EŞİK: {main_threshold:.4f}")
print("="*50)

# --- 7. TEST RAPORU ---
test_cluster_ids = kmeans.predict(X_test_scaled)
y_pred = [cluster_map[c] for c in test_cluster_ids]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nTest Seti Doğruluğu: %{acc*100:.2f}")
print(f"Genel F1 Skoru     : %{f1*100:.2f}")

print("\n--- KARMAŞIKLIK MATRİSİ (Confusion Matrix) ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- AYRINTILI SINIFLANDIRMA RAPORU ---")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Saldırı'], digits=4))

# =====================================================
# --- 7B. TRAIN (KDDTrain+) DOĞRULUK ---
# =====================================================
print("\n==================== TRAIN SONUÇLARI ====================")
train_pred = [cluster_map[c] for c in train_clusters]

train_acc = accuracy_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)

print(f"Eğitim Seti Doğruluğu: %{train_acc*100:.2f}")
print(f"Eğitim F1 Skoru      : %{train_f1*100:.2f}")

print("\n--- TRAIN Confusion Matrix ---")
print(confusion_matrix(y_train, train_pred))

print("\n--- TRAIN Classification Report ---")
print(classification_report(y_train, train_pred, target_names=['Normal', 'Saldırı'], digits=4))

# =====================================================
# --- EK 4: IDS Mode Threshold Kullanarak Test ---
# =====================================================
print("\n[*] Threshold ile IDS Mod Test Çalıştırılıyor...")

test_distances = kmeans.transform(X_test_scaled)
test_min_dists = np.min(test_distances, axis=1)

y_pred_threshold = []

for dist, cid in zip(test_min_dists, test_cluster_ids):
    if dist > main_threshold:
        y_pred_threshold.append(1)  # anomali
    else:
        y_pred_threshold.append(cluster_map[cid])

print("\n--- Threshold Bazlı Sonuçlar ---")
print("Accuracy:", accuracy_score(y_test, y_pred_threshold))
print("F1:", f1_score(y_test, y_pred_threshold))
print(confusion_matrix(y_test, y_pred_threshold))

# =====================================================
# --- EK 5: Sonuçları Kaydet ---
# =====================================================
results_df = pd.DataFrame({
    "true_label": y_test,
    "predicted_cluster_map": y_pred,
    "predicted_threshold_mode": y_pred_threshold
})

results_df.to_csv("test_results.csv", index=False)
print("\n[+] Test sonuçları 'test_results.csv' olarak kaydedildi.")

# --- 8. KAYDETME ---
save_data = {'map': cluster_map, 'threshold': main_threshold}

print("\n[*] Modeller kaydediliyor...")
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(save_data, 'threshold.pkl')
print("[+] İŞLEM TAMAMLANDI. (Ayrıntılı rapor + analizler oluşturuldu.)")
