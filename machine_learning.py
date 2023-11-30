import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
import time

# Bu fonksiyon,  output.csv dosyasından veri setini çeker
def Build_Data_Set():
    
    df = pd.read_csv('output.csv', index_col=0)   
    df_train = df[:50000]  # Son 50000 satırı eğitim veri seti olarak kullan
    df_test = df[10000:]  # İlk 10000 satırı test veri seti olarak kullan

    # Eğitim veri setini temizle ve ölçeklendir
    X_train = np.array(df_train.drop(['classification', 'usage_counter', 'normal_prio', 'policy', 'vm_pgoff', 'task_size', 'cached_hole_size', 'hiwater_rss', 'nr_ptes', 'lock', 'cgtime', 'signal_nvcsw'], 1))
    X_train = preprocessing.scale(X_train)
    # Eğitim etiketi
    y_train = np.array(df_train['classification'].replace("malware", 0).replace("benign", 1))

    # Test veri setini temizle ve ölçeklendir
    X_test = np.array(df_test.drop(['classification', 'usage_counter', 'normal_prio', 'policy', 'vm_pgoff', 'task_size', 'cached_hole_size', 'hiwater_rss', 'nr_ptes', 'lock', 'cgtime', 'signal_nvcsw'], 1))
    X_test = preprocessing.scale(X_test)
    # Test etiketi
    y_test = np.array(df_test['classification'].replace("malware", 0).replace("benign", 1))

    return X_train, X_test, y_train, y_test  # Dizileri döndür

# Bu fonksiyon, kendi veri setinizle bir makine öğrenimi modeli oluşturur ve doğruluk oranını hesaplar
def Analysis():
    test_size = 10000  # Test veri setinin boyutu
    X_train, X_test, y_train, y_test = Build_Data_Set()  # Eğitim ve test veri setlerini oluştur

    clf = svm.SVC(kernel="linear", C=0.01)  # Scikit-learn paketi ile bir svm nesnesi oluştur
    clf.fit(X_train, y_train)  # Modeli eğitim veri seti ve etiketleriyle eğit
    result = clf.predict(X_test)  # 10000 örnek üzerinde bir tahmin çalıştır

    # Doğruluk oranını hesapla ve yazdır
    correct_count = 0
    for i in range(0, test_size):
        if result[i] == y_test[i]:
            correct_count += 1

    print("Accuracy:", (correct_count / test_size) * 100)
    return

start = time.time()
Analysis()  # Programı çalıştır
end = time.time()

elapsed = end - start

print("Time:", elapsed)

