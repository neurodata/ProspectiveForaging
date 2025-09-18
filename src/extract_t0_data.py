import numpy as np

# Extract T=0 data and combine with original results
for rep in range(10):
    # Load T=0 file (has 9 rows: T=0 + 8 other T values)
    t0_file = f'../results/PAFM/NN/NN_onehot_w_lookahead_t0_reps{rep}.npz'
    t0_data = np.load(t0_file)['future_states']

    # Load original file (has 8 rows: T=[128,256,512,750,1024,1250,1500,2048])
    orig_file = f'../results/PAFM/NN/NN_onehot_w_lookahead_reps{rep}.npz'
    orig_data = np.load(orig_file)['future_states']

    # T=0 is the first row in the T=0 file
    t0_baseline = t0_data[0:1]  # Shape (1, 101)

    # Combine: T=0 first, then original T values
    combined = np.vstack([t0_baseline, orig_data])  # Shape (9, 101)

    # Save combined results
    output_file = f'../results/PAFM/NN/NN_onehot_w_lookahead_with_t0_reps{rep}.npz'
    np.savez(output_file, future_states=combined)

    print(f"Rep {rep}: Combined T=0 + original data -> {output_file}")
    print(f"  T=0 shape: {t0_baseline.shape}, Original shape: {orig_data.shape}")
    print(f"  Combined shape: {combined.shape}")

print("\nExtraction complete! Now you have T=0 properly positioned as the first row.")