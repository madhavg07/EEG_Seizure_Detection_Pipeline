import os
import re
import mne
import numpy as np
import scipy.signal as signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit
import matplotlib.pyplot as plt
import warnings
import gc  

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# --- CONFIGURATION ---
PATIENTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb08', 'chb10', 'chb11', 'chb24']
WINDOW_SEC = 2
SLIDE_SEC = 1.0  
MIN_SEIZURE_DURATION = 8  
COLLAR_SEC = 30           

# --- HELPER FUNCTIONS ---
def get_seizure_times(pid, filename):
    summary_path = os.path.join(pid, f"{pid}-summary.txt")
    if not os.path.exists(summary_path): return None, None
    with open(summary_path, 'r') as f: text = f.read()
    blocks = text.split(f"File Name: {filename}")
    if len(blocks) > 1:
        block = blocks[1].split("File Name:")[0]
        start = re.search(r'Seizure\s*\d*\s*Start Time:\s*(\d+)', block)
        end = re.search(r'Seizure\s*\d*\s*End Time:\s*(\d+)', block)
        if start and end: return int(start.group(1)), int(end.group(1))
    return None, None

def generate_sharp_spike(sfreq, target_freq):
    n = int(sfreq / target_freq)
    x = np.linspace(0, 1, n)
    y = np.where(x <= 0.4, x * 15/4, x * (-5)/2 + 5/2)
    y = y / target_freq
    if np.std(y) > 0: y = (y - np.mean(y)) / np.std(y) / np.sqrt(n)
    return y

def build_clinical_dictionary(sfreq, window_samples):
    dictionary = []
    target_frequencies = [5, 10, 15, 20, 24] 
    for freq in target_frequencies:
        spike = generate_sharp_spike(sfreq, freq)
        padded_spike = np.zeros(window_samples)
        if len(spike) < window_samples:
            start_idx = (window_samples - len(spike)) // 2
            padded_spike[start_idx:start_idx+len(spike)] = spike
            dictionary.append(padded_spike)
    return np.array(dictionary).T if dictionary else np.zeros((window_samples, 5))

# --- FEATURE EXTRACTION ---
def extract_window_features(window, sfreq):
    window_samples = window.shape[1]
    clinical_dict = build_clinical_dictionary(sfreq, window_samples)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=2) 
    features = []
    for ch in window:
        if np.var(ch) < 1e-10 or clinical_dict.shape[0] != window_samples:
            features.extend([0.0] * 10)
            continue
        omp.fit(clinical_dict, ch)
        features.extend(np.abs(omp.coef_))
        variance = np.var(ch)
        line_length = np.sum(np.abs(np.diff(ch)))
        features.extend([variance, line_length])
        
        nperseg = min(len(ch), int(sfreq)) 
        if nperseg < 1:
            features.extend([0.0, 0.0, 0.0])
        else:
            freqs, psd = signal.welch(ch, sfreq, nperseg=nperseg)
            theta_power = np.sum(psd[(freqs >= 4) & (freqs < 8)])
            alpha_power = np.sum(psd[(freqs >= 8) & (freqs < 13)])
            beta_power  = np.sum(psd[(freqs >= 13) & (freqs < 30)])
            features.extend([theta_power, alpha_power, beta_power])
        
    corr_matrix = np.corrcoef(window)
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    correlations = corr_matrix[upper_tri_indices]
    correlations = np.nan_to_num(correlations)
    features.extend(correlations)
    return features

def fixed_segmentation(components, sfreq):
    window_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(SLIDE_SEC * sfreq)
    n_windows = (components.shape[1] - window_samples) // step_samples + 1
    features, window_times = [], []
    for i in range(max(1, n_windows)):
        start = i * step_samples
        end = start + window_samples
        features.append(extract_window_features(components[:, start:end], sfreq))
        window_times.append((start / sfreq, end / sfreq))
    return np.array(features), window_times

def adaptive_segmentation(components, sfreq):
    total_samples = components.shape[1]
    
    # MEMORY FIX: Replaced the memory-heavy Hilbert FFT with a lightweight absolute envelope.
    envelope = np.abs(components[0, :]) 
    
    rolling_var = np.convolve(envelope, np.ones(int(sfreq))/int(sfreq), mode='same')
    variance_changes = np.abs(np.gradient(rolling_var))
    threshold = np.mean(variance_changes) + (2 * np.std(variance_changes))
    change_points, _ = signal.find_peaks(variance_changes, height=threshold, distance=int(sfreq*0.5))
    boundaries = [0] + list(change_points) + [total_samples]
    
    features, window_times = [], []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if (end - start) < (sfreq * 0.5): continue
        features.append(extract_window_features(components[:, start:end], sfreq))
        window_times.append((start / sfreq, end / sfreq))
        
    return np.array(features), window_times

def extract_events(binary_sequence, slide_sec, m_dur=0):
    events = []
    in_event = False
    start_idx = 0
    for i, val in enumerate(binary_sequence):
        if val == 1 and not in_event:
            start_idx = i
            in_event = True
        elif val == 0 and in_event:
            end_idx = i
            duration = (end_idx - start_idx) * slide_sec
            if duration >= m_dur:
                events.append((start_idx * slide_sec, end_idx * slide_sec))
            in_event = False
    if in_event:
        duration = (len(binary_sequence) - start_idx) * slide_sec
        if duration >= m_dur:
            events.append((start_idx * slide_sec, len(binary_sequence) * slide_sec))
    return events

def calculate_paper_metrics(y_true, y_pred, slide_sec, total_hours):
    true_events = extract_events(y_true, slide_sec, m_dur=0)
    pred_events = extract_events(y_pred, slide_sec, m_dur=MIN_SEIZURE_DURATION)
    tp, fn, fp = 0, 0, 0

    for t_start, t_end in true_events:
        matched = False
        for p_start, p_end in pred_events:
            if max(t_start - COLLAR_SEC, p_start) <= min(t_end + COLLAR_SEC, p_end):
                matched = True
                break
        if matched: tp += 1
        else: fn += 1

    for p_start, p_end in pred_events:
        matched = False
        for t_start, t_end in true_events:
            if max(t_start - COLLAR_SEC, p_start) <= min(t_end + COLLAR_SEC, p_end):
                matched = True
                break
        if not matched: fp += 1

    total_windows = int((total_hours * 3600) / slide_sec)
    tn = total_windows - (tp + fp + fn)
    if tn < 0: tn = 0

    sens = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
    acc = ((tp + tn) / (tp + tn + fp + fn)) * 100 if (tp + tn + fp + fn) > 0 else 0.0
    
    # THE CRITICAL FIX: FDE is "Failure to Detect Events" (Miss Rate)
    fde = (fn / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    
    # THE DISCOVERY: FDR exposes the Precision flaw
    fdr = (fp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    
    far = fp / total_hours if total_hours > 0 else 0.0

    return tp, tn, fp, fn, sens, spec, acc, fde, fdr, far

# --- CUSTOM SCICA ALGORITHM ---
def scica_sc_update(hc, hcref, maxangle=np.pi/180):
    hcreturn = hc.copy()
    for i in range(hcref.shape[1]):
        a = hc[:, i]
        b = hcref[:, i]
        theta = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
        if np.abs(theta) < maxangle:
            hcreturn[:, i] = a
        else:
            ab = float(np.dot(b, a))
            c1 = (1 - 2*ab + ab**2) - ((2 - 2*ab) * (np.cos(maxangle))**2)
            c2 = (2*ab - 2*ab**2) - ((2*ab - 2) * (np.cos(maxangle))**2)
            c3 = ab**2 - (np.cos(maxangle))**2
            temp = np.sqrt(max(c2**2 - 4*c1*c3, 0)) 
            x1 = (-c2 + temp) / (2*c1) if c1 != 0 else 0
            x2 = (-c2 - temp) / (2*c1) if c1 != 0 else 0
            y = (1 - x1)*a + x1*b if 0 < x1 < 1 else (1 - x2)*a + x2*b
            hcreturn[:, i] = y / np.linalg.norm(y)
    return hcreturn

def run_scica(X, n_comp, SC=None, sc_typ='soft', alpha=1.0, maxit=200, tol=1e-4):
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    p = X_centered.shape[1]
    V = (X_centered @ X_centered.T) / p
    u, s, _ = np.linalg.svd(V)
    D = np.diag(1.0 / np.sqrt(s[:n_comp]))
    K = D @ u[:, :n_comp].T
    X1 = K @ X_centered
    
    if SC is None:
        SCnum = 0
        w_init = np.random.randn(n_comp, n_comp)
    else:
        SCnum = SC.shape[0]
        q, _ = np.linalg.qr((K @ SC.T))
        wc = q
        w_init_rand = np.random.randn(n_comp, n_comp - wc.shape[1])
        w_init = np.hstack((wc, w_init_rand))
        q_w, _ = np.linalg.qr(w_init)
        w_init = q_w.T
        
    W = w_init.T
    hcref = W[:, :SCnum] if SCnum > 0 else None
    W1 = W.copy()
    lim = 1000
    it = 0
    while lim > tol and it < maxit:
        if SCnum == 0 or sc_typ == 'weak':
            wx = W @ X1
            gwx = np.tanh(alpha * wx)
            v1 = (gwx @ X1.T) / p
            g_wx = alpha * (1 - gwx**2)
            v2 = np.diag(np.mean(g_wx, axis=1)) @ W
            W1 = v1 - v2
            u_w, d_w, _ = np.linalg.svd(W1)
            W1 = u_w @ np.diag(1.0/d_w) @ u_w.T @ W1
        else:
            hc = W[:, :SCnum]
            hu = W[:, SCnum:]
            wx = hu.T @ X1
            gwx = np.tanh(alpha * wx)
            v1 = (gwx @ X1.T).T / p
            g_wx = alpha * (1 - gwx**2)
            v2 = hu @ np.diag(np.mean(g_wx, axis=1))
            hu1 = v1 - v2
            if sc_typ == 'soft':
                W1_temp = np.hstack((hu1, hc))
                q_w, _ = np.linalg.qr(W1_temp)
                hc = q_w[:, -SCnum:]
                hu1 = q_w[:, :-SCnum]
                hc = scica_sc_update(hc, hcref)
            W1 = np.hstack((hc, hu1))
            q_w, _ = np.linalg.qr(W1)
            W1 = q_w
        lim = np.max(np.abs(np.abs(np.diag(W1 @ W.T)) - 1))
        W = W1
        it += 1
    return W.T @ K

# --- OPTIMAL THRESHOLDING (From Paper Section VI) ---
def find_optimal_threshold(clf, X_train, y_train):
    """Dynamically finds the best threshold maximizing 30*TPR - FPR as per the paper."""
    probs = clf.predict_proba(X_train)[:, 1]
    best_score = -np.inf
    best_t = 0.50
    
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (np.array(y_train) == 1))
        fp = np.sum((preds == 1) & (np.array(y_train) == 0))
        fn = np.sum((preds == 0) & (np.array(y_train) == 1))
        tn = np.sum((preds == 0) & (np.array(y_train) == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # The exact mathematical objective function used by the original authors 
        score = 30 * tpr - fpr
        
        if score > best_score:
            best_score = score
            best_t = t
            
    return best_t


# ==========================================
# --- PART 3: MASTER EXECUTION LOOP ---
# ==========================================
final_results_table = []
plot_data_f = [] 

for patient in PATIENTS:
    print(f"\n\n=======================================================")
    print(f"=== PROCESSING PATIENT: {patient} ===")
    print(f"=======================================================")
    
    record_file = os.path.join(patient, "local_records.txt")
    if not os.path.exists(record_file):
        print(f"Skipping {patient}: Make sure local_records.txt exists in the folder.")
        continue
        
    with open(record_file, 'r') as f:
        target_files = f.read().splitlines()

    is_trained = False
    fixed_unmixing_matrix = None
    best_thresh_f = 0.50
    best_thresh_a = 0.50

    clf_fixed = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, class_weight={0: 1, 1: 5}, n_jobs=-1, random_state=42)
    clf_adaptive = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, class_weight={0: 1, 1: 5}, n_jobs=-1, random_state=42)

    true_fix, pred_fix = [], []
    true_ada, pred_ada = [], []
    hours_tested = 0.0

    for filename in target_files:
        file_path = os.path.join(patient, filename)
        if not os.path.exists(file_path):
            print(f" -> Missing file: {file_path}")
            continue
            
        print(f"\n--- Reading {filename} ---")
        start_sec, end_sec = get_seizure_times(patient, filename)
        has_seizure = (start_sec is not None)
        
        raw_info = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        sfreq = raw_info.info['sfreq']
        max_time = (raw_info.n_times - 1) / sfreq
        del raw_info
        gc.collect()
        
        # --- PHASE 1: CALIBRATION ---
        if not is_trained:
            if has_seizure:
                raw_train = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
                n_channels = len(raw_train.ch_names)
                SC_matrix = np.zeros((1, n_channels))
                SC_matrix[0, 5:8] = 1.0 
                
                print(" -> Running Custom SCICA Optimization (Memory-Optimized)...")
                buffer_sec = 300 
                tmin = max(0, start_sec - buffer_sec)
                tmax = min(max_time, end_sec + buffer_sec)
                
                raw_train.crop(tmin=tmin, tmax=tmax)
                raw_train.load_data(verbose=False) 
                raw_train.filter(l_freq=0.53, h_freq=70.0, fir_design='firwin', verbose=False)
                raw_train.notch_filter(freqs=60.0, verbose=False)
                
                scica_training_data = raw_train.get_data()
                fixed_unmixing_matrix = run_scica(X=scica_training_data, n_comp=10, SC=SC_matrix, sc_typ='soft')
                train_components = np.dot(fixed_unmixing_matrix, scica_training_data)
                
                adj_start_sec = start_sec - tmin
                adj_end_sec = end_sec - tmin
                
                print(" -> Extracting Features & Dynamically Optimizing Thresholds (Paper Section VI)...")
                X_train_f, times_f = fixed_segmentation(train_components, sfreq)
                y_train_f = [1 if (e > adj_start_sec and s < adj_end_sec) else 0 for s, e in times_f]
                clf_fixed.fit(X_train_f, y_train_f)
                best_thresh_f = find_optimal_threshold(clf_fixed, X_train_f, y_train_f)
                
                X_train_a, times_a = adaptive_segmentation(train_components, sfreq)
                y_train_a = [1 if (e > adj_start_sec and s < adj_end_sec) else 0 for s, e in times_a]
                clf_adaptive.fit(X_train_a, y_train_a)
                best_thresh_a = find_optimal_threshold(clf_adaptive, X_train_a, y_train_a)
                
                is_trained = True
                print(f" -> [CALIBRATED] Locked Thresholds -> Fixed: {best_thresh_f:.2f} | Adaptive: {best_thresh_a:.2f}")
                del raw_train, scica_training_data, train_components
                gc.collect()
            else:
                print(" -> Standby: Normal rhythm. Waiting for training seizure...")

        # --- PHASE 2: AUTONOMOUS MONITORING ---
        else:
            TARGET_SMOOTH_SEC = 20.0 
            SMOOTH_WINDOW_FIXED = int(TARGET_SMOOTH_SEC / SLIDE_SEC)
            SMOOTH_WINDOW_ADA = int(TARGET_SMOOTH_SEC / WINDOW_SEC) 
            REQUIRED_SEIZURE_SEC = 5.0 
            chunk_duration = 300  # Strict 5-minute memory chunking
            
            for chk_st in np.arange(0, max_time, chunk_duration):
                chk_en = min(chk_st + chunk_duration, max_time)
                if chk_en - chk_st < 5: continue 
                
                print(f"    -> Analyzing time chunk: {chk_st/60:.0f}m to {chk_en/60:.0f}m...")
                raw_chk = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
                raw_chk.crop(tmin=chk_st, tmax=chk_en)
                raw_chk.load_data(verbose=False)
                raw_chk.filter(l_freq=0.53, h_freq=70.0, fir_design='firwin', verbose=False)
                raw_chk.notch_filter(freqs=60.0, verbose=False)
                
                test_components = np.dot(fixed_unmixing_matrix, raw_chk.get_data())
                
                # 1. Test Fixed Model
                X_test_f, times_f = fixed_segmentation(test_components, sfreq)
                if len(X_test_f) > 0:
                    raw_probs_f = clf_fixed.predict_proba(X_test_f)[:, 1]
                    smoothed_probs_f = np.convolve(raw_probs_f, np.ones(SMOOTH_WINDOW_FIXED)/SMOOTH_WINDOW_FIXED, mode='same')
                    base_preds_f = (smoothed_probs_f >= best_thresh_f).astype(int)
                    
                    req_windows_f = int(REQUIRED_SEIZURE_SEC / SLIDE_SEC)
                    preds_f = np.zeros_like(base_preds_f)
                    for i in range(req_windows_f - 1, len(base_preds_f)):
                        if np.sum(base_preds_f[i-req_windows_f+1:i+1]) == req_windows_f:
                            preds_f[i] = 1
                            
                    pred_fix.extend(preds_f)
                    true_array = [1 if (start_sec is not None and (e+chk_st) > start_sec and (s+chk_st) < end_sec) else 0 for s, e in times_f]
                    true_fix.extend(true_array)
                    
                    # Save probability traces for plotting (only if seizure exists in this chunk)
                    if sum(true_array) > 0:
                        plot_data_f.append({'patient': patient, 'probs': smoothed_probs_f, 'trues': true_array, 'thresh': best_thresh_f})
                
                # 2. Test Adaptive Model
                X_test_a, times_a = adaptive_segmentation(test_components, sfreq)
                if len(X_test_a) > 0:
                    raw_probs_a = clf_adaptive.predict_proba(X_test_a)[:, 1]
                    smoothed_probs_a = np.convolve(raw_probs_a, np.ones(SMOOTH_WINDOW_ADA)/SMOOTH_WINDOW_ADA, mode='same')
                    base_preds_a = (smoothed_probs_a >= best_thresh_a).astype(int)
                    
                    preds_a = np.zeros_like(base_preds_a)
                    consecutive_count = 0
                    for i in range(len(base_preds_a)):
                        if base_preds_a[i] == 1:
                            consecutive_count += 1
                            elapsed_sec = consecutive_count * WINDOW_SEC
                            if elapsed_sec >= REQUIRED_SEIZURE_SEC:
                                preds_a[i] = 1
                        else:
                            consecutive_count = 0
                            
                    pred_ada.extend(preds_a)
                    true_ada.extend([1 if (start_sec is not None and (e+chk_st) > start_sec and (s+chk_st) < end_sec) else 0 for s, e in times_a])
                
                del raw_chk, test_components
                gc.collect()

            print(" -> Data processed. Event arrays updated.")
            hours_tested += (max_time / 3600.0)

    # Collect Patient Results for Global Table
    if hours_tested > 0:
        f_tp, f_tn, f_fp, f_fn, f_sens, f_spec, f_acc, f_fde, f_fdr, f_far = calculate_paper_metrics(true_fix, pred_fix, SLIDE_SEC, hours_tested)
        a_tp, a_tn, a_fp, a_fn, a_sens, a_spec, a_acc, a_fde, a_fdr, a_far = calculate_paper_metrics(true_ada, pred_ada, WINDOW_SEC, hours_tested)
        
        final_results_table.append({
            'Patient': patient, 'Hours': round(hours_tested, 2),
            'ADA_SENS': round(a_sens, 2), 'ADA_SPEC': round(a_spec, 2), 'ADA_ACC': round(a_acc, 2),
            'ADA_FDE': round(a_fde, 2), 'ADA_FDR': round(a_fdr, 2), 'ADA_FAR': round(a_far, 2),
            'FIX_SENS': round(f_sens, 2), 'FIX_SPEC': round(f_spec, 2), 'FIX_ACC': round(f_acc, 2),
            'FIX_FDE': round(f_fde, 2), 'FIX_FDR': round(f_fdr, 2), 'FIX_FAR': round(f_far, 2) 
        })

# ==========================================
# --- PART 4: FINAL CONSOLIDATED RESULTS ---
# ==========================================
print("\n\n" + "="*160)
print("=== FINAL CONSOLIDATED RESULTS TABLE (ADAPTIVE VS FIXED SPLITTING) ===")
print("="*160)
print(f"{'Patient':<10} | {'Hrs':<6} || {'ADA SENS':<8} | {'ADA SPEC':<8} | {'ADA ACC':<7} | {'ADA FDE':<7} | {'ADA FDR':<7} | {'ADA FAR':<7} || {'FIX SENS':<8} | {'FIX SPEC':<8} | {'FIX ACC':<7} | {'FIX FDE':<7} | {'FIX FDR':<7} | {'FIX FAR':<7}")
print("-" * 160)
for r in final_results_table:
    print(f"{r['Patient']:<10} | {r['Hours']:<6} || {r['ADA_SENS']:>8.2f} | {r['ADA_SPEC']:>8.2f} | {r['ADA_ACC']:>7.2f} | {r['ADA_FDE']:>7.2f} | {r['ADA_FDR']:>7.2f} | {r['ADA_FAR']:>7.2f} || {r['FIX_SENS']:>8.2f} | {r['FIX_SPEC']:>8.2f} | {r['FIX_ACC']:>7.2f} | {r['FIX_FDE']:>7.2f} | {r['FIX_FDR']:>7.2f} | {r['FIX_FAR']:>7.2f}")
print("="*160)

# --- PLOTTING DIRECT OUTPUT (SIMULATING PAPER FIG 4) ---
if len(plot_data_f) > 0:
    print(f"\nGenerating and saving {len(plot_data_f)} Probabilistic Seizure Plots...")
    
    for index, sample_plot in enumerate(plot_data_f):
        plt.figure(figsize=(12, 5))
        
        time_axis = np.arange(len(sample_plot['probs'])) * SLIDE_SEC
        plt.plot(time_axis, sample_plot['probs'], label="Random Forest Probability", color="blue", linewidth=1.5)
        plt.axhline(y=sample_plot['thresh'], color='red', linestyle='--', label=f"Optimal Dynamic Threshold ({sample_plot['thresh']:.2f})")
        
        # Shade True Seizure Regions
        in_seizure = False
        start_shade = 0
        for i, val in enumerate(sample_plot['trues']):
            if val == 1 and not in_seizure:
                start_shade = i * SLIDE_SEC
                in_seizure = True
            elif val == 0 and in_seizure:
                label = "Actual Seizure Onset" if start_shade == 0 else ""
                plt.axvspan(start_shade, i * SLIDE_SEC, color='red', alpha=0.2, label=label)
                in_seizure = False
                
        plt.title(f"Seizure Detection Probability Trace ({sample_plot['patient']})")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Seizure Probability")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"seizure_plot_{sample_plot['patient']}_{index}.png"
        plt.savefig(filename, dpi=300)
        plt.close() 
        print(f" -> Saved {filename}")
        
    print("\nAll plots saved successfully! Check your folder to grab them for your presentation.")