import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Определение 20 реалистичных шагов воронки
steps = [
    'landing_page',
    'homepage',
    'category_browse',
    'search',
    'product_view',
    'product_detail',
    'add_to_wishlist',
    'add_to_cart',
    'cart_view',
    'checkout_start',
    'shipping_info',
    'payment_method',
    'promo_code',
    'order_review',
    'purchase_complete',
    'exit_browse',
    'exit_cart',
    'exit_checkout',
    'support_chat',
    'return_policy'
]

# Генерация данных
np.random.seed(42)
data = []
start_time = datetime(2024, 1, 1, 8, 0, 0)

for user_id in range(1, 1001):
    # Определяем тип пользователя (разные сценарии поведения)
    user_type = np.random.choice(
        ['browser', 'cart_abandoner', 'checkout_abandoner', 'buyer', 'quick_exit'],
        p=[0.35, 0.30, 0.20, 0.10, 0.05]
    )

    # Начинаем всегда с лендинга или хоумпейджа
    current_time = start_time + timedelta(
        minutes=np.random.randint(0, 60 * 24 * 30),  # случайная дата в течение месяца
        seconds=np.random.randint(0, 60)
    )

    # Выбор начального шага
    path = ['landing_page']
    current_time += timedelta(seconds=np.random.randint(5, 30))

    # Генерация пути в зависимости от типа пользователя
    if user_type == 'quick_exit':
        # Быстрый уход после лендинга
        path.append('exit_browse')
        steps_count = 2

    elif user_type == 'browser':
        # Просматривает товары но не добавляет в корзину
        steps_count = np.random.randint(3, 8)
        possible_steps = ['homepage', 'category_browse', 'search', 'product_view', 'product_detail']
        for _ in range(steps_count - 1):
            path.append(np.random.choice(possible_steps))
        path.append('exit_browse')
        steps_count += 1

    elif user_type == 'cart_abandoner':
        # Добавляет в корзину но не покупает
        path.extend(['homepage', 'category_browse', 'product_view', 'product_detail', 'add_to_cart', 'cart_view'])
        # Иногда возвращается к просмотру
        if np.random.random() > 0.6:
            path.extend(['product_view', 'product_detail'])
        path.append('exit_cart')
        steps_count = len(path)

    elif user_type == 'checkout_abandoner':
        # Доходит до чекаута но бросает
        path.extend(['homepage', 'category_browse', 'search', 'product_view',
                     'product_detail', 'add_to_cart', 'cart_view', 'checkout_start',
                     'shipping_info', 'payment_method'])
        # Иногда смотрит промокод или политику возврата
        if np.random.random() > 0.5:
            path.append('promo_code')
        if np.random.random() > 0.7:
            path.append('return_policy')
        path.append('exit_checkout')
        steps_count = len(path)

    elif user_type == 'buyer':
        # Совершает покупку (иногда с циклами)
        path.extend(['homepage', 'category_browse', 'product_view', 'product_detail'])
        # Цикл: возврат к поиску
        if np.random.random() > 0.4:
            path.extend(['search', 'product_view'])
        path.extend(['add_to_cart', 'cart_view', 'checkout_start', 'shipping_info',
                     'payment_method', 'order_review', 'purchase_complete'])
        steps_count = len(path)

    # Добавляем случайные задержки между шагами и записываем в данные
    for i, step in enumerate(path):
        # Добавляем небольшой шум в шаги для разнообразия
        if i > 0 and np.random.random() > 0.9 and len(path) > 5:
            extra_step = np.random.choice(['support_chat', 'search', 'category_browse'])
            data.append({
                'user_id': user_id,
                'step': extra_step,
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
            })
            current_time += timedelta(seconds=np.random.randint(10, 45))

        data.append({
            'user_id': user_id,
            'step': step,
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
        })

        # Время между шагами зависит от типа шага
        if 'exit' in step or step == 'purchase_complete':
            delay = np.random.randint(5, 20)
        elif 'product' in step or step == 'search':
            delay = np.random.randint(20, 120)
        elif 'cart' in step or 'checkout' in step:
            delay = np.random.randint(30, 180)
        else:
            delay = np.random.randint(5, 45)

        current_time += timedelta(seconds=delay)

# Создаем DataFrame
df = pd.DataFrame(data)

# Проверка: убедимся что все 20 шагов присутствуют
actual_steps = sorted(df['step'].unique())
print(f"Уникальных шагов в данных: {len(actual_steps)}")
print(f"Шаги: {actual_steps}")
print(f"\nВсего записей: {len(df)}")
print(f"Уникальных пользователей: {df['user_id'].nunique()}")

# Сохраняем в CSV
df.to_csv('user_journey_1000_users_20_steps.csv', index=False)
print("\n✅ Файл успешно сохранен: user_journey_1000_users_20_steps.csv")

# Показываем пример данных
print("\nПример первых 20 записей:")
print(df.head(20).to_string(index=False))

# Статистика по шагам
print("\nСтатистика по шагам:")
step_stats = df['step'].value_counts().head(10)
print(step_stats)