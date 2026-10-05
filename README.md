# Gerçek Zamanlı Akıllı Trafik Yoğunluk Tahmin Sistemi

Bu sistem, Lambda Mimarisi prensiplerine uygun olarak geliştirilmiş uçtan uca bir veri hattıdır (data pipeline).

## Temel Teknolojiler ve Mimari
* **Data Ingestion:** TomTom API'den çekilen canlı verilerin **Apache Kafka** ile sisteme aktarılması.
* **Batch Processing:** **Apache Spark** ile geçmiş verilerin işlenip **MongoDB**'ye kaydedilmesi.
* **Streaming Processing:** **Spark Streaming** ile akan verinin anlık işlenmesi ve **Redis** üzerinde önbelleklenmesi.
* **Machine Learning:** **XGBoost** ile zaman serisi mantığında gelecekteki trafik hacminin tahmin edilmesi ve anomali tespiti.
* **Serving Layer:** **Flask API** ve **Streamlit** ile anlık yoğunluk skorlarının ve makine öğrenmesi tahminlerinin görselleştirilmesi.

## Çalıştırma
Sistemi izole bir ortamda ayağa kaldırmak için:
`docker-compose -f docker.yml up -d`
