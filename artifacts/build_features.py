"""
Модуль с функцией build_features для сборки признакового пространства.

Используется при внедрении модели: читает CSV, объединяет их по user_id,
генерирует признаки, возвращает датафрейм на уровне пользователя.

При необходимости замените пути на актуальные.
"""

import pandas as pd
import numpy as np

# ========== константы путей ==========
USERS_PATH = '/datasets/ds_s13_users.csv'
VISITS_PATH = '/datasets/ds_s13_visits.csv'
ADS_ACTIVITY_PATH = '/datasets/ads_activity.csv'
SURF_DEPTH_PATH = '/datasets/surf_depth.csv'
PRIMARY_DEVICE_PATH = '/datasets/primary_device.csv'
CLOUD_USAGE_PATH = '/datasets/cloud_usage.csv'

# ========== константы проекта ==========
TARGET_COL = 'age_category'
USER_ID_COL = 'user_id'


def build_features(
    users_path=USERS_PATH,
    visits_path=VISITS_PATH,
    ads_activity_path=ADS_ACTIVITY_PATH,
    surf_depth_path=SURF_DEPTH_PATH,
    primary_device_path=PRIMARY_DEVICE_PATH,
    cloud_usage_path=CLOUD_USAGE_PATH,
    target_col=TARGET_COL,
    user_id_col=USER_ID_COL,
):    
    # 1. Читаем CSV
    users = pd.read_csv(users_path)
    visits = pd.read_csv(visits_path)
    ads_activity = pd.read_csv(ads_activity_path)
    surf_depth = pd.read_csv(surf_depth_path)
    primary_device = pd.read_csv(primary_device_path)
    cloud_usage = pd.read_csv(cloud_usage_path)

    # 2. Чистим от дубликатов по user_id
    users = users.drop_duplicates(subset=[user_id_col], keep="first")
    ads_activity = ads_activity.drop_duplicates(subset=[user_id_col], keep="first")

    # 3. Приводим типы
    visits["date"] = pd.to_datetime(visits["date"])
    cloud_usage["cloud_usage"] = cloud_usage["cloud_usage"].astype(int)

    # 4. Признаки на основе visits
    # 4.1. Агрегаты по сессиям
    visits_agg = visits.groupby(user_id_col).agg(
        n_sessions=("session_id", "nunique"),
        n_categories=("website_category", "nunique"),
        n_visits=("session_id", "size"),
        n_days=("date", "nunique"),
    ).reset_index()

    # 4.2. Среднее число сессий в день
    visits_agg["sessions_per_day"] = visits_agg["n_sessions"] / visits_agg["n_days"]

    # 4.3. Доли активности по категориям сайтов
    pivot_cat = pd.pivot_table(
        visits,
        index=user_id_col,
        columns="website_category",
        values="session_id",
        aggfunc="count",
        fill_value=0,
    )
    pivot_cat_share = pivot_cat.div(pivot_cat.sum(axis=1), axis=0)
    pivot_cat_share.columns = [
        f"share_cat_{c.replace('Category ', '').lower()}"
        for c in pivot_cat_share.columns
    ]
    pivot_cat_share = pivot_cat_share.reset_index()

    # 4.4. Доли активности по времени суток
    daytime_map = {
        "утро": "morning",
        "день": "day",
        "вечер": "evening",
        "ночь": "night",
    }
    pivot_daytime = pd.pivot_table(
        visits,
        index=user_id_col,
        columns="daytime",
        values="session_id",
        aggfunc="count",
        fill_value=0,
    )
    pivot_daytime_share = pivot_daytime.div(pivot_daytime.sum(axis=1), axis=0)
    pivot_daytime_share.columns = [
        f"share_daytime_{daytime_map[c]}" for c in pivot_daytime_share.columns
    ]
    pivot_daytime_share = pivot_daytime_share.reset_index()

    # 4.5. Наиболее активное время суток
    most_active_daytime = (
        pivot_daytime.idxmax(axis=1)
        .map(daytime_map)
        .rename("most_active_daytime")
        .reset_index()
    )

    # 4.6. Доля активности в выходные
    visits["is_weekend"] = visits["date"].dt.dayofweek.isin([5, 6]).astype(int)
    weekend_share = (
        visits.groupby(user_id_col)["is_weekend"]
        .mean()
        .rename("share_weekend")
        .reset_index()
    )

    # 5. Объединяем всё по user_id
    features = users[[user_id_col, target_col]].copy()

    for df in [
        visits_agg,
        pivot_cat_share,
        pivot_daytime_share,
        most_active_daytime,
        weekend_share,
        ads_activity,
        surf_depth,
        primary_device,
        cloud_usage,
    ]:
        features = features.merge(df, on=user_id_col, how="left")

    return features
