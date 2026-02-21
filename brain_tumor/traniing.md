📁 Creating directory structure...                                                   │
│ ✅ Directories created!                                                              │
│                                                                                      │
│ ======================================================================               │
│ 🧹 STARTING DATASET CLEANING                                                         │
│ ======================================================================               │
│                                                                                      │
│ 📊 Processing Training data...                                                       │
│                                                                                      │
│   Processing glioma: 1321 images                                                     │
│   glioma: 100%|██████████████████████████| 1321/1321 [00:04<00:00, 309.42it/s]       │
│   ✅ Valid: 1321, ❌ Corrupted: 0                                                    │
│                                                                                      │
│   Processing meningioma: 1339 images                                                 │
│   meningioma: 100%|██████████████████████| 1339/1339 [00:04<00:00, 304.52it/s]       │
│   ✅ Valid: 1339, ❌ Corrupted: 0                                                    │
│                                                                                      │
│   Processing notumor: 1595 images                                                    │
│   notumor: 100%|█████████████████████████| 1595/1595 [00:03<00:00, 429.14it/s]       │
│   ✅ Valid: 1595, ❌ Corrupted: 0                                                    │
│                                                                                      │
│   Processing pituitary: 1457 images                                                  │
│   pituitary: 100%|███████████████████████| 1457/1457 [00:05<00:00, 288.24it/s]       │
│   ✅ Valid: 1457, ❌ Corrupted: 0                                                    │
│                                                                                      │
│ 📊 Processing Testing data...                                                        │
│                                                                                      │
│   Processing glioma: 300 images                                                      │
│   glioma: 100%|████████████████████████████| 300/300 [00:00<00:00, 322.27it/s]       │
│   ✅ Valid: 300, ❌ Corrupted: 0                                                     │
│                                                                                      │
│   Processing meningioma: 306 images                                                  │
│   meningioma: 100%|████████████████████████| 306/306 [00:00<00:00, 356.52it/s]       │
│   ✅ Valid: 306, ❌ Corrupted: 0                                                     │
│                                                                                      │
│   Processing notumor: 405 images                                                     │
│   notumor: 100%|███████████████████████████| 405/405 [00:00<00:00, 547.83it/s]       │
│   ✅ Valid: 405, ❌ Corrupted: 0                                                     │
│                                                                                      │
│   Processing pituitary: 300 images                                                   │
│   pituitary: 100%|█████████████████████████| 300/300 [00:00<00:00, 310.96it/s]       │
│   ✅ Valid: 300, ❌ Corrupted: 0                                                     │
│                                                                                      │
│ ======================================================================               │
│ 📊 CLEANING REPORT                                                                   │
│ ======================================================================               │
│ Total images processed: 7023                                                         │
│ Successfully cleaned: 7023                                                           │
│ Corrupted/Invalid: 0                                                                 │
│ Success rate: 100.00%                                                                │
│                                                                                      │
│ 📈 Class Distribution:                                                               │
│   Testing_glioma: 300 images                                                         │
│   Testing_meningioma: 306 images                                                     │
│   Testing_notumor: 405 images                                                        │
│   Testing_pituitary: 300 images                                                      │
│   Training_glioma: 1321 images                                                       │
│   Training_meningioma: 1339 images                                                   │
│   Training_notumor: 1595 images                                                      │
│   Training_pituitary: 1457 images                                                    │
│                                                                        



 95   # Training augmentation - conservative for medical images                                        │
│  96   train_datagen = ImageDataGenerator(                                                              │
│  97       rescale=1./255,                                                                              │
│  98 -     rotation_range=15,                                                                           │
│  99 -     width_shift_range=0.1,                                                                       │
│ 100 -     height_shift_range=0.1,                                                                      │
│ 101 -     shear_range=0.1,                                                                             │
│ 102 -     zoom_range=0.1,                                                                              │
│  98 +     rotation_range=20,                                                                           │
│  99 +     width_shift_range=0.2,                                                                       │
│ 100 +     height_shift_range=0.2,                                                                      │
│ 101 +     shear_range=0.2,                                                                             │
│ 102 +     zoom_range=0.2,                                                                              │
│ 103       horizontal_flip=True,                                                                        │
│ 104 -     brightness_range=[0.85, 1.15],                                                               │
│ 104 +     vertical_flip=True,                                                                          │
│ 105 +     brightness_range=[0.8, 1.2],                                                                 │
│ 106       fill_mode='constant',                                                                        │
│ 107       cval=0,                                                                                      │
│ 108       validation_split=config.VALIDATION_SPLI