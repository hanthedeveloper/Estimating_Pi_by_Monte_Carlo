# Estimating Pi by Monte Carlo (`pimc`)

Monte Carlo yöntemiyle π tahmini: **seri** ve **paralel** (`multiprocessing.Pool`) sürümler, bu sürümleri karşılaştıran deney scriptleri ve sonuçları çizen analiz scriptleri.

Yöntem: birim karede rastgele `n` nokta seçilir, `x² + y² ≤ 1` olan (çeyrek daireye düşen) noktalar sayılır ve `π ≈ 4 · inside / n` hesaplanır.

Projenin sorduğu sorular:

- Hata `n` büyüdükçe nasıl azalıyor? (beklenen: `O(1/√n)`)
- İşçi sayısı `p` arttıkça ne kadar hızlanıyoruz? (speedup, efficiency)
- NumPy ve saf Python neden farklı ölçekleniyor? (chunk boyutu, önbellek, bellek bant genişliği)
- Küçük `n` için paralel hangi noktadan sonra değiyor?
- İş, işçi sayısıyla birlikte büyütülürse süre sabit kalıyor mu? (weak scaling)

---

## İçindekiler

1. [Proje yapısı](#1-proje-yapısı)
2. [Kurulum](#2-kurulum)
3. [Kullanım](#3-kullanım)
4. [Sonuç dosyalarını okuma](#4-sonuç-dosyalarını-okuma)
5. [Notlar ve sınırlılıklar](#5-notlar-ve-sınırlılıklar)

---

## 1. Proje yapısı

```
Estimating_Pi_by_Monte_Carlo/
├── pyproject.toml          # Poetry ayarları, bağımlılıklar
├── poetry.lock
├── README.md
├── notes.txt               # rapor içindeki bölüm planı ve çalışma notları
│
├── src/pimc/               # kütüphane (import pimc...)
│   ├── base.py             # RunConfig, EstimateResult, PiEstimator (soyut sınıf)
│   ├── serial.py           # count_pure, count_numpy, PureSerialEstimator, NumpySerialEstimator
│   ├── parallel.py         # ParallelEstimator (Pool tabanlı)
│   ├── benchmark.py        # run_benchmark: bir tahmin edici listesini tekrarlı çalıştırıp CSV'ye yazar
│   ├── io_utils.py         # CSV'ye satır ekleme (append_row)
│   └── metrics.py          # speedup, efficiency, abs_error
│
├── scripts/                # DENEYLER: ölçüm alır, results/*.csv üretir
│   ├── run_accuracy.py
│   ├── run_speedup.py
│   ├── run_chunk.py
│   ├── run_smalln.py
│   └── run_weak.py
│
├── analyze_accuracy.py     # ANALİZ: results/*.csv okur, tablo yazdırır, results/*.png üretir
├── analyze_speedup.py
├── analyze_chunk.py
├── analyze_smalln.py
├── analyze_weak.py
│
├── try_*.py                # hızlı deneme scriptleri (rapora girmeyen ön denemeler)
├── tests/                  # test_estimators.py (şu an boş)
└── results/                # deney çıktıları (CSV) ve grafikler (PNG)
```

İş akışı iki adımdır:

```
scripts/run_*.py  ──►  results/*.csv  ──►  analyze_*.py  ──►  results/*.png + terminalde tablo
   (ölçüm)              (ham veri)           (analiz)
```

---

## 2. Kurulum

Gereksinimler: Python ≥ 3.11 (ölçümler Python 3.13.1 ile alındı) ve [Poetry](https://python-poetry.org/).

```bash
poetry install
```

Bu komut `pimc` paketini ve bağımlılıkları (`numpy`, `pandas`, `matplotlib`, `pydantic`) kurar.

> **Önemli:** Tüm komutları **proje kök klasöründen** çalıştırın. Scriptler `results/...` yolunu göreli yazar; başka klasörden çalıştırırsanız dosyalar yanlış yere yazılır veya bulunamaz.

---

## 3. Kullanım

### 3.1 Kütüphane olarak

```python
from pimc.serial import NumpySerialEstimator, PureSerialEstimator
from pimc.parallel import ParallelEstimator

# Seri (NumPy)
r = NumpySerialEstimator().estimate(n=10_000_000, seed=1)
print(r.pi_estimate, r.error, r.elapsed)

# Paralel (4 işçi, NumPy)
r = ParallelEstimator(workers=4, method="numpy").estimate(n=100_000_000, seed=1)
print(r.workers, r.pi_estimate, r.elapsed)
```

`estimate()` bir `EstimateResult` döndürür: `estimator`, `n`, `workers`, `inside`, `elapsed` (saniye), `pi_estimate` ve `error` (= `|pi_estimate − π|`).

| Sınıf | Açıklama |
|---|---|
| `PureSerialEstimator` | Saf Python döngüsü, seri. Referans (en yavaş) taban çizgisi. |
| `NumpySerialEstimator(chunk=...)` | NumPy vektörleştirilmiş, seri. Hızlı taban çizgisi. |
| `ParallelEstimator(workers=p, method="numpy" \| "pure", chunk=...)` | `n` noktayı `p` işçi sürece böler, kısmi sayımları toplar. |

`chunk`: NumPy'ın bir seferde ürettiği nokta sayısıdır (varsayılan `1_000_000`). Bellek kullanımını ve önbellek davranışını belirler; sonuçlarda önemli bir etkisi vardır (bkz. bölüm 4).

### 3.2 Deneyleri çalıştırma

```bash
poetry run python scripts/run_accuracy.py
poetry run python scripts/run_speedup.py
poetry run python scripts/run_chunk.py
poetry run python scripts/run_smalln.py
poetry run python scripts/run_weak.py
```

| Script | Ne ölçer | Çıktı |
|---|---|---|
| `run_accuracy.py` | `n = 10³ … 10⁸`, NumPy seri ve 4 işçili paralel, 20 tekrar | `results/accuracy.csv` |
| `run_speedup.py` | A: NumPy, `n = 10⁸`, p ∈ {1,2,4,6,8,12}, 10 tekrar. B: saf Python, `n = 10⁷`, aynı p'ler, 5 tekrar | `results/speedup_numpy.csv`, `results/speedup_pure.csv` |
| `run_chunk.py` | NumPy, `n = 10⁸`, chunk ∈ {10⁶, 10⁵, 10⁴}, p ∈ {1,2,4,6,12}, 5 tekrar | `results/chunk_1000000.csv`, `chunk_100000.csv`, `chunk_10000.csv` |
| `run_smalln.py` | chunk = 10⁴, `n = 10³ … 10⁸`, p ∈ {2,4,6,12}, 10 tekrar | `results/smalln.csv` |
| `run_weak.py` | chunk = 10⁴, işçi başına 10⁷ nokta (`n = p · 10⁷`), p ∈ {1,2,4,6,12}, 10 tekrar | `results/weak.csv` |

Uyarılar:

- **CSV dosyaları üzerine yazılmaz, sonuna eklenir.** Bir deneyi tekrar çalıştırırsanız satırlar ikiye katlanır ve medyan/ortalama bozulur. Yeniden çalıştırmadan önce ilgili CSV'yi silin.
- Ölçümler uzun sürebilir (özellikle `n = 10⁸` ve saf Python). Her ölçüm hemen dosyaya yazıldığı için deney yarıda kesilse de o ana kadarki satırlar korunur.
- Bir ölçüm hata verirse mesaj yazdırılır, o ölçüm atlanır ve deney devam eder.
- Ölçüm sırasında bilgisayarda başka ağır işlem çalıştırmayın; zamanlamaları bozar.

### 3.3 Analiz ve grafikler

CSV'ler hazır olduktan sonra:

```bash
poetry run python analyze_accuracy.py   # -> results/accuracy.png
poetry run python analyze_speedup.py    # -> results/speedup.png
poetry run python analyze_chunk.py      # -> results/speedup_chunk.png
poetry run python analyze_smalln.py     # -> results/smalln.png
poetry run python analyze_weak.py       # -> results/weak.png
```

Her script özet tabloyu terminale yazdırır ve grafiği kaydeder. Eşleşmeler:

| Analiz scripti | Okuduğu CSV | Ürettiği grafik |
|---|---|---|
| `analyze_accuracy.py` | `accuracy.csv` | `accuracy.png` |
| `analyze_speedup.py` | `speedup_numpy.csv`, `speedup_pure.csv` | `speedup.png` |
| `analyze_chunk.py` | `chunk_*.csv`, `speedup_pure.csv` | `speedup_chunk.png` |
| `analyze_smalln.py` | `smalln.csv` | `smalln.png` |
| `analyze_weak.py` | `weak.csv` | `weak.png` |

`analyze_chunk.py`, `analyze_speedup.py` içindeki `make_table` fonksiyonunu kullanır; bu yüzden iki dosya aynı klasörde kalmalıdır.

`try_*.py` dosyaları rapora girmeyen hızlı denemelerdir (ör. `try_bench.py` `results/try.csv` yazar). Normal akışta çalıştırmanız gerekmez.

---

## 4. Sonuç dosyalarını okuma

### 4.1 CSV formatı

Tüm CSV dosyaları aynı sütunlara sahiptir. **Her satır tek bir ölçümdür.**

| Sütun | Anlamı |
|---|---|
| `estimator` | `pure_serial`, `numpy_serial`, `parallel_pure`, `parallel_numpy` |
| `n` | Toplam nokta sayısı (tüm işçiler toplamı) |
| `workers` | İşçi süreç sayısı `p` (seri tahmin edicilerde 1) |
| `repeat` | Tekrar numarası (0'dan başlar) |
| `seed` | Bu ölçümde kullanılan tohum: `1000 + repeat` |
| `elapsed` | Sayım adımının duvar saati süresi, saniye |
| `pi_estimate` | `4 · inside / n` |
| `error` | `abs(pi_estimate − π)` (mutlak hata) |

Dikkat edilecekler:

- **Süre ölçümü:** `elapsed` yalnızca nokta sayma adımını içerir. `Pool` oluşturma ve kapatma süresi **dahil değildir**.
- **Tekrarlanabilirlik:** Aynı `(estimator, n, workers, chunk, seed)` için `pi_estimate` ve `error` her çalıştırmada aynıdır. `elapsed` ise makineye ve anlık yüke göre değişir.
- **Seri ve `p=1` paralel farkı:** `numpy_serial` ana süreçte, `parallel_numpy` ile `workers=1` ise yeni açılan bir işçi süreçte çalışır. İkisi aynı kodu çalıştırsa da süreleri birebir aynı çıkmayabilir.
- **Chunk değeri CSV'de yazmaz.** Hangi chunk ile ölçüldüğü dosya adından ve ilgili `run_*.py` scriptinden anlaşılır (aşağıdaki tabloya bakın).

### 4.2 Her CSV dosyası ne anlatıyor?

| Dosya | Satır | Estimator'lar | n | p | chunk | Neyi cevaplar |
|---|---|---|---|---|---|---|
| `accuracy.csv` | 240 | `numpy_serial`, `parallel_numpy` | 10³ … 10⁸ | 1, 4 | 10⁶ | Hata `n` ile nasıl azalıyor? Seri ve paralel aynı doğrulukta mı? |
| `speedup_numpy.csv` | 70 | `numpy_serial`, `parallel_numpy` | 10⁸ | 1, 2, 4, 6, 8, 12 | 10⁶ | NumPy ile speedup ve efficiency |
| `speedup_pure.csv` | 35 | `pure_serial`, `parallel_pure` | 10⁷ | 1, 2, 4, 6, 8, 12 | yok | Saf Python ile speedup ve efficiency |
| `chunk_1000000.csv` | 30 | `numpy_serial`, `parallel_numpy` | 10⁸ | 1, 2, 4, 6, 12 | 10⁶ | Chunk'ın ölçeklenmeye etkisi |
| `chunk_100000.csv` | 30 | aynı | 10⁸ | 1, 2, 4, 6, 12 | 10⁵ | aynı |
| `chunk_10000.csv` | 30 | aynı | 10⁸ | 1, 2, 4, 6, 12 | 10⁴ | aynı |
| `smalln.csv` | 300 | `numpy_serial`, `parallel_numpy` | 10³ … 10⁸ | 2, 4, 6, 12 | 10⁴ | Paralel hangi `n`'den sonra seriden hızlı? |
| `weak.csv` | 60 | `numpy_serial` (yalnızca p=1), `parallel_numpy` | `p · 10⁷` | 1, 2, 4, 6, 12 | 10⁴ | İş, işçiyle birlikte büyüyünce süre sabit kalıyor mu? |

### 4.3 CSV'yi kendiniz okumak

```python
import pandas as pd

df = pd.read_csv("results/speedup_numpy.csv")

# Süre için ortalama değil MEDYAN kullanın (tek tük yavaş ölçümlerden etkilenmez)
med = df.groupby(["estimator", "workers"])["elapsed"].median()

t_serial = med[("numpy_serial", 1)]
t_p4     = med[("parallel_numpy", 4)]
print("speedup(p=4) =", t_serial / t_p4)

# Hata için tekrarlar üzerinden ortalama ve standart sapma
acc = pd.read_csv("results/accuracy.csv")
print(acc.groupby(["estimator", "n"])["error"].agg(["mean", "std"]))
```

### 4.4 Metrikler

| Metrik | Formül | Yorum |
|---|---|---|
| Speedup | `S = T_seri / T_paralel` | 1'den büyükse paralel daha hızlı. İdeal değer `p`. |
| Efficiency | `E = S / p` | 1 = mükemmel. Düşük değer: ek yük, paylaşılan kaynak (bellek/önbellek) çekişmesi veya yük dengesizliği. |
| Weak scaling verimliliği | `T(1) / T(p)` | İşçi başına iş sabitken süre sabit kalırsa 1. |
| Scaled speedup | `p · T(1) / T(p)` | Weak scaling'de ideal değer `p`. |

Hesaplamalarda seri taban çizgisi, **aynı CSV içindeki seri ölçümün medyanıdır** (aynı `n`, aynı chunk). Farklı dosyalardaki seri süreleri birbirine karıştırmayın.

`efficiency > 1` çıkabilir. Bu fiziksel bir üstünlük değil, çoğunlukla ölçüm gürültüsü veya seri taban çizgisinin ana süreçte biraz yavaş ölçülmesidir. Tekrar sayısı az olduğunda (5) medyan gürültüye daha açıktır.

### 4.5 Grafikleri okuma

**`accuracy.png`** (doğruluk ve yakınsama)
- Eksenler: x = `n`, y = ortalama mutlak hata. İkisi de logaritmik.
- Çizgiler: `numpy_serial`, `parallel_numpy` (her nokta 20 tekrarın ortalaması) ve siyah kesikli teorik eğri `0.8·σ`, burada `σ = 4·√(p̂(1−p̂)/n)`.
- Okuma: hata doğrusal bir düşüş gösteriyorsa ve eğim yaklaşık **−0,5** ise hata `O(1/√n)` ile azalıyordur. İki tahmin edici çizgisinin birbirine ve teori çizgisine yakın olması seri ve paralel sürümlerin aynı doğrulukta olduğunu gösterir. Küçük `n` değerlerinde tekrarlar arası dağılım büyüktür, çizgi bu bölgede dalgalanabilir.

**`speedup.png`** (speedup, NumPy ve saf Python)
- Eksenler: x = işçi sayısı `p`, y = speedup.
- Çizgiler: NumPy (`n = 10⁸`), saf Python (`n = 10⁷`) ve siyah kesikli ideal çizgi `S = p`. Dikey noktalı çizgi 6 fiziksel çekirdeği işaret eder.
- Okuma: ideal çizgiye ne kadar yakınsa o kadar verimli. Eğrinin ideal çizgiden koptuğu `p` değeri ölçeklenmenin tıkandığı yerdir. Fiziksel çekirdek sayısını (dikey çizgi) aşan `p` değerlerinde artış küçülür, hatta düşebilir: ek işçiler hyper-threading ile aynı çekirdeği paylaşır.

**`speedup_chunk.png`** (chunk etkisi)
- Eksenler: x = `p`, y = speedup (her çizgi kendi chunk'lı seri ölçümüne göre).
- Çizgiler: NumPy için üç chunk değeri (10⁶, 10⁵, 10⁴), gri kesikli saf Python ve ideal çizgi.
- Okuma: aynı kod, aynı makine, yalnızca chunk değişiyor. Chunk küçüldükçe eğri idealine yaklaşıyorsa, zayıf ölçeklenmenin nedeni algoritma değil çalışma kümesinin önbelleğe sığıp sığmamasıdır.

**`smalln.png`** (küçük `n` ve başlangıç maliyeti)
- Eksenler: x = `n` (log), y = speedup (log).
- Çizgiler: her `p` için bir çizgi ve siyah kesikli `speedup = 1` (başabaş çizgisi).
- Okuma: çizgi başabaş çizgisinin **altındaysa paralel seriden yavaştır**. Çizginin 1'i kestiği `n`, paralelin işe yarmaya başladığı eşiktir. Küçük `n`'de süreç açma ve veri gönderme maliyeti işin kendisinden büyüktür.

**`weak.png`** (weak scaling, iki panel)
- Sol panel: x = `p`, y = süre. Nokta medyan, hata çubuğu min–max aralığıdır. Siyah kesikli çizgi ideal durumu (sabit süre) gösterir.
- Sağ panel: x = `p`, y = `T(1)/T(p)`. Siyah kesikli çizgi 1'dir (ideal).
- Okuma: süre sabit kalıyor, verimlilik 1'e yakınsa iyi ölçeklenme vardır. Hata çubukları genişse o noktadaki ölçümler dağınıktır. Bu da arka plan yükü veya bellek baskısından kaynaklanabilir.

### 4.6 Terminalde yazdırılan tablolar

`analyze_speedup.py` ve `analyze_chunk.py` şu sütunlarla bir tablo yazdırır: `workers`, `median_s` (medyan süre), `speedup`, `efficiency`. Üst satırda ilgili dosyanın seri medyan süresi görünür.
`analyze_smalln.py` satırı `n`, sütunu `p` olan bir speedup matrisi yazdırır.
`analyze_weak.py` her `p` için `n`, `median_s`, `efficiency` ve `scaled_speedup` yazdırır.
`analyze_accuracy.py` her `(estimator, n)` için ortalama ve standart sapma hatayı yazdırır.

---

## 5. Notlar ve sınırlılıklar

- **Sonuçlar makineye özgüdür.** Mevcut ölçümler Intel Core i5-10500 (6 fiziksel çekirdek, 12 mantıksal çekirdek) üzerinde alınmıştır. Başka bir işlemcide speedup eğrileri ve chunk etkisi farklı çıkabilir. Kendi makinenizde karşılaştırma yapmak için deneyleri yeniden çalıştırın.
- **Dışarıdaki etkiler:** Ölçümler sırasında başka süreçler açıktı ve bellek baskısı olmuş olabilir. Bu, özellikle `p = 6` ve `p = 12`'deki tekrarlar arası dağılımı artırabilir.
- **Önbellek açıklaması bir hipotezdir.** Chunk ve bellek bant genişliği yorumu ölçümlerle *tutarlıdır*, ancak önbellek isabet oranı doğrudan ölçülmemiştir (`perf` vb. kullanılmadı).
- **Çoklu işlem başlatma yöntemi:** Python 3.13'te Linux varsayılanı `fork`'tur. Python 3.14 ile varsayılan değişirse zamanlamalar etkilenebilir.
- **Rastgele sayı akışları bağımsızdır:** Her işçi `SeedSequence(seed).spawn(p)` ile ayrı bir tohum alır. Chunk değiştirmek rastgele akışın bölünme biçimini değiştirir, bu yüzden `pi_estimate` değerleri biraz farklı çıkabilir. İkisi de geçerli Monte Carlo tahminleridir.
- `notes.txt`: rapor bölüm planı ve ölçümler üzerine çalışma notlarını içerir. Çalışma için gerekli değildir.
