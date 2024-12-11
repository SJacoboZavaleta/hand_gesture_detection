model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )