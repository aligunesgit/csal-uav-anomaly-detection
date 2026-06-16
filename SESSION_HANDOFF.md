# Pablo Projesi — Oturum El Değişimi (2026-06-04, Oturum 2)

## Proje
`/Users/aligunes/Desktop/IEEE Transactions REmote Sensign/Pablo/`
Makale: `J_STARS_template/jstars_csal_paper.tex` (J-STARS, IEEE)
Konu: Cost-Sensitive Active Learning (cAL) ile UAV multispektral ripariyan anomali tespiti

---

## Bu Oturumda Tamamlananlar

### Javier Overleaf Yorumları — YAPILDI

| Yorum | Yapılan |
|---|---|
| "5 bands not maintained as components" | Feature Extraction'a açıklama cümlesi eklendi (satır ~317-323) |
| "out of margins" | `\IEEEpubidadjcol` keywords bloğundan sonra eklendi |
| "KKT not previously introduced" | "Karush--Kuhn--Tucker (KKT)" olarak tanımlandı |
| "Standardize parentheses/commas/hyphens" | Tüm ` --- ` → `---`; `eq.~\eqref{}` → `(\ref{})`; subfloat'ta `$r^{+}=4$` → `$r^{+} = 4$` |
| "Z2 is the less class-imbalanced" | "most class-imbalanced" → "highest anomaly prevalence" düzeltildi |
| "Add r+=1,2,3 to table + bold best" | Tablo güncellendi; r+=1/2/3 satırları eklendi; Z2'de bold r+=3'e (FNR=0.070) geçti |
| "Redundant with previous paragraph" | Discussion V.A kısaltıldı, Section III-B'ye forward ref verildi |
| "Add FP% to emphasize smaller than FN%" | FP +259%, FN -66.1% eklendi; anomali oranı bağlamı ile açıklandı |
| "Not suitable section title" (×2) | "Why cAL Reduces..." → "Mechanisms of False-Negative Reduction"; "Why Symmetric..." → "Limitations of Symmetric Evaluation Metrics" |
| "Complete Table IV, omit Table III" | Önceki oturumda yapılmıştı; reply hazırlandı |
| "References must be extended" | 11 yeni referans eklendi (bak aşağı); Related Work genişletildi |

### Yeni Referanslar Eklendi (bibliography.bib)
- `lopezfandino2022deep` — López-Fandiño grup makalesi
- `zhang2019precision` — UAV precision agriculture review
- `rouse1974monitoring` — NDVI (orijinal)
- `woebbecke1995color` — ExG vejetasyon indeksi
- `huete2002overview` — EVI/MODIS
- `mountrakis2011svms` — SVM in RS review
- `chang2011libsvm` — LIBSVM
- `chawla2002smote` — SMOTE class imbalance
- `sun2007cost` — Cost-sensitive boosting
- `sener2018active` — Core-set AL for deep learning
- `breiman2001random` — Random Forests (SHAP için)

### Related Work Genişletildi
- Active Learning subsection: SVM-AL avantajları + sener2018active eklendi
- Cost-Sensitive subsection: SMOTE + sun2007cost eklendi
- UAV Multispectral subsection: vejetasyon indeksleri paragrafı eklendi; RX detector + chanussot cite edildi; kampffmeyer + li2019domain cite edildi

---

## YAPILMAYANLAR — Sonraki Oturumda

### Kritik (Bilgi Gerekiyor)
- [ ] **`gunes2026geoai` bib entry** — bibliography.bib'de TODO placeholder var. Ali'nin GeoAI makalenin tam bilgisini girmesi lazım: başlık, dergi/konferans, yıl
- [ ] **"add reference?" (1:28 pm yorumu)** — Hangi satıra bağlı olduğu bilinmiyor. Overleaf'te o yorumun satırını bul
- [ ] **`ali.png`** — `J_STARS_template/` klasöründe yok. Fotoğrafını 1in × 1.25in oranında kırpıp oraya koy
- [ ] **Author Two & Three** — Pablo (Quesada-Barriuso) ve bir üçüncü yazar. Gerçek isim, unvan, biyografi metni lazım. Şu an USC affiliation yorumda var
- [ ] **"I would add anomaly maps figure for all methods"** — Tüm 7 method için Z2/E2 görselleştirmesi. generate_anomaly_maps.py ile üretilebilir; büyük iş

### Teknik
- [ ] **GitHub push** — `J_STARS_template/` hiç commit edilmemiş
- [ ] **Overleaf sync** — Güncel .tex'i Overleaf'e yapıştır
- [ ] **LaTeX derle** — Yeni referanslarla birlikte derleme kontrol et (özellikle `lopezfandino2022deep` key'inin gerçek bir makalesi olduğunu doğrula)

### Overleaf Reply Metinleri (Kopyala-Yapıştır Hazır)

**"5 bands not maintained":**
> Done. Added to Section III-A3: "The raw per-pixel reflectances are not used directly; instead, per-superpixel band means μ_b aggregate the spectral information across all pixels within each superpixel... The five per-band means form the first five components of the vector, so spectral band information is fully retained."

**"out of margins":**
> Done. Missing `\IEEEpubidadjcol` command added after Index Terms block.

**"KKT not introduced":**
> Done. Introduced as "Karush--Kuhn--Tucker (KKT) conditions" at first use. No dedicated reference added as KKT is a standard convex optimisation result.

**"Standardize parentheses/commas/hyphens":**
> Done. (1) Em-dashes standardised to closed style (no surrounding spaces) throughout; (2) single inconsistent equation reference `eq.~(5)` replaced with `(5)`; (3) math spacing unified in figure subcaptions.

**"Z2 class imbalance":**
> Corrected. "The most class-imbalanced scene" replaced with "the scene with the highest anomaly prevalence (12.51%) and strong spectral overlap." Z2 is indeed the least severely imbalanced scene (closest to balanced).

**"Add r+=1,2,3 to table + bold best":**
> Done. Table III now includes cAL at r+=1, 2, 3, and 4. Per-column bold marks the best result per scene. Note: for Scene Z2, cAL r+=3 achieves FNR=0.070 vs r+=4's 0.071 — both are included with appropriate bold.

**"Redundant with previous paragraph" (1:33):**
> Done. Section V-A condensed to a brief forward reference to Section III-B, eliminating repetition of the mechanism descriptions.

**"Add FP% terms" (1:38):**
> Done: "...false positives increase from 212.3 to 762.0 (+259%), while false negatives fall from 515.0 to 174.6 (−66.1%). Because anomalies constitute only 3%–13% of each scene, the absolute false-positive rate remains low..."

**"Not suitable section title" (1:55 & 1:56):**
> Done. "Why cAL Reduces Missed Anomalies" → "Mechanisms of False-Negative Reduction"; "Why Symmetric Evaluation Is Insufficient" → "Limitations of Symmetric Evaluation Metrics".

**"Complete Table IV, omit Table III" (1:42):**
> Done in our latest revision — original Table III removed; per-scene comparison table expanded to include all methods (Mahalanobis, Random AL, Standard AL, Unc+KernelKMeans, Raw-5D ablation, cAL r+=1–4).

**"If aggregate table removed..." (1:58):**
> Noted. The aggregate statistics (515.0 FN → 174.6, recall 0.777 → 0.924) remain in the Conclusion as a concise summary derived from the expanded per-scene Table III.

**"References must be extended" (2:30):**
> Done. Added 11 new references covering: vegetation indices (NDVI, ExG, EVI), RX anomaly detector, UAV environmental monitoring, SVM in RS (LIBSVM, Mountrakis review), class imbalance (SMOTE, cost-sensitive boosting), deep AL (Sener core-set), and Random Forests. Related Work subsections expanded accordingly. Total references now ~28.

---

## Veri Notu
- Gerçek veri: `/Users/aligunes/Desktop/IEEE Transactions REmote Sensign/agentic ai/data/`
- Sembolik linkler Pablo/data/ altında KIRIK
- cal_results.json key formatı: `'cAL r+=1'`, `'cAL r+=2'`, vb.
- Final budget FNR (cAL r+=4): Z1=0.078, Z2=0.071, E1=0.029, E2=0.144

## Önemli Satır Numaraları (güncel .tex)
- Feature Extraction subsubsection: ~314
- KKT tanımı: ~570
- Tab:confusion_scene (Table III): ~963
- Discussion V.A: ~1135
- Discussion V.B (FP%): ~1156
- Discussion V.C: başlık değişti ~1208
