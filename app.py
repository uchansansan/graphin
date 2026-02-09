import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import os
import tempfile
from datetime import datetime

# =============== СТИЛИЗАЦИЯ ===============
st.set_page_config(
    page_title='Графин',
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_validate_data(uploaded_file):
    """Загрузка и валидация CSV файла"""
    try:
        df = pd.read_csv(uploaded_file)

        # Автоопределение колонок
        required_cols = ['user_id', 'step', 'timestamp']
        found_cols = {}

        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['user_id', 'user', 'userid', 'id', 'идентификатор', 'пользователь']:
                found_cols['user_id'] = col
            elif col_lower in ['step', 'stage', 'event', 'action', 'label', 'метка', 'шаг', 'этап', 'действие']:
                found_cols['step'] = col
            elif col_lower in ['timestamp', 'time', 'date', 'datetime', 'время', 'дата']:
                found_cols['timestamp'] = col

        if len(found_cols) < 3:
            return None, f"Нужны колонки: user_id, step, timestamp. Найдено: {list(found_cols.keys())}"

        # Переименовываем в стандартные имена
        df = df.rename(columns={
            found_cols['user_id']: 'user_id',
            found_cols['step']: 'step',
            found_cols['timestamp']: 'timestamp'
        })

        # Конвертируем timestamp
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            return None, f"Не удалось распознать формат даты: {str(e)}"

        return df, "OK"
    except Exception as e:
        return None, f"Ошибка загрузки: {str(e)}"

def prepare_transitions(df):
    """Подготовка данных: сортировка, переходы, циклы (исправлено!)"""
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    df['next_step'] = df.groupby('user_id')['step'].shift(-1)
    transitions = df.dropna(subset=['next_step']).copy()

    # Исправленное обнаружение циклов (без проблем с индексами)
    transitions['is_cycle'] = False

    # Для каждого пользователя отслеживаем историю посещённых шагов
    for user_id, group in transitions.groupby('user_id', group_keys=False):
        seen_steps = set()
        indices = group.index.tolist()
        steps = group['step'].tolist()
        next_steps = group['next_step'].tolist()

        for i, idx in enumerate(indices):
            current_step = steps[i]
            next_step = next_steps[i]

            # Если следующий шаг уже был посещён ранее — это цикл
            if next_step in seen_steps:
                transitions.at[idx, 'is_cycle'] = True

            # Добавляем текущий шаг в историю
            seen_steps.add(current_step)

    return transitions

def build_graph_pyvis(transitions, min_weight=1, max_weight=9999, selected_nodes=None):
    """Создание интерактивного графа Pyvis с фильтром по выбранным нодам"""
    # Агрегация переходов с информацией о циклах
    edges = transitions.groupby(['step', 'next_step']).agg(
        count=('user_id', 'count'),
        cycle_count=('is_cycle', 'sum')
    ).reset_index()
    edges['is_cycle_edge'] = edges['cycle_count'] > 0

    # Фильтрация по весу
    edges = edges[(edges['count'] >= min_weight) & (edges['count'] <= max_weight)]

    # Фильтрация по выбранным нодам
    if selected_nodes and len(selected_nodes) > 0:
        edges = edges[
            (edges['step'].isin(selected_nodes)) |
            (edges['next_step'].isin(selected_nodes))
        ]

    if edges.empty:
        return None, "Нет данных после фильтрации"

    # Уникальные шаги (только те, что остались после фильтрации)
    all_steps = pd.concat([edges['step'], edges['next_step']]).unique()

    # Создаём граф
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
        notebook=False
    )

    # # Добавляем узлы
    # step_visits = pd.concat([edges['step'], edges['next_step']]).value_counts()
    # for step in all_steps:
    #     size = 15 + min(45, step_visits.get(step, 0) / 1.5)
    #     is_highlight = selected_nodes and step in selected_nodes
    #
    #     # Цвета узлов
    #     step_lower = str(step).lower()
    #     if 'purchase' in step_lower or 'success' in step_lower or 'complete' in step_lower or 'оплата' in step_lower:
    #         color = '#27ae60'
    #     elif 'exit' in step_lower or 'abandon' in step_lower or 'drop' in step_lower or 'leave' in step_lower or 'выход' in step_lower:
    #         color = '#e74c3c'
    #     elif 'cart' in step_lower or 'checkout' in step_lower or 'payment' in step_lower or 'корзина' in step_lower:
    #         color = '#f39c12'
    #     elif is_highlight:
    #         color = '#9b59b6'
    #     else:
    #         color = '#3498db'
    #
    #     net.add_node(
    #         step,
    #         label=str(step),
    #         title=f"Шаг: {step}\nПосещений: {step_visits.get(step, 0)}",
    #         size=size,
    #         color=color,
    #         font={'size': 14, 'color': 'white' if is_highlight else 'black'}
    #     )
    #
    # # Добавляем рёбра
    # for _, row in edges.iterrows():
    #     src, tgt, cnt, is_cycle = row['step'], row['next_step'], row['count'], row['is_cycle_edge']
    #
    #     color = '#e67e22' if is_cycle else '#3498db'
    #     width = max(1.5, min(12, cnt / 2))
    #     dashes = [5, 5] if is_cycle else False
    #
    #     net.add_edge(
    #         str(src),
    #         str(tgt),
    #         value=cnt,
    #         title=f"{src} → {tgt}\nПереходов: {cnt}" + (" [ЦИКЛ]" if is_cycle else ""),
    #         width=width,
    #         color=color,
    #         dashes=dashes,
    #         smooth={'type': 'curvedCW', 'roundness': 0.4}
    #     )

    # Добавляем узлы
    step_visits = pd.concat([edges['step'], edges['next_step']]).value_counts()

    # Определяем тупиковые узлы (из которых не выходят рёбра)
    outgoing_edges = edges.groupby('step').size()
    sink_nodes = [step for step in all_steps if step not in outgoing_edges.index]

    # Находим узлы с максимальным количеством исходящих рёбер
    if not outgoing_edges.empty:
        max_outgoing = outgoing_edges.max()
        max_outgoing_nodes = outgoing_edges[outgoing_edges == max_outgoing].index.tolist()
    else:
        max_outgoing_nodes = []
        max_outgoing = 0

    for step in all_steps:
        size = 15 + min(45, step_visits.get(step, 0) / 1.5)
        is_highlight = selected_nodes and step in selected_nodes

        # Цвета узлов по новой логике
        if step in sink_nodes:
            # Красный для тупиковых узлов (нет исходящих рёбер)
            color = '#e74c3c'
            title_suffix = " ⚠️ ТУПИКОВЫЙ"
        elif step in max_outgoing_nodes:
            # Оранжевый для узлов с макс. исходящими рёбрами
            color = '#f39c12'
            title_suffix = f" 🌟 МАКС. ИСХОДЯЩИХ ({max_outgoing})"
        else:
            # Стандартные цвета для остальных
            step_lower = str(step).lower()
            if 'purchase' in step_lower or 'success' in step_lower or 'complete' in step_lower or 'оплата' in step_lower:
                color = '#27ae60'
            elif 'exit' in step_lower or 'abandon' in step_lower or 'drop' in step_lower or 'leave' in step_lower or 'выход' in step_lower:
                color = '#e74c3c'
            elif 'cart' in step_lower or 'checkout' in step_lower or 'payment' in step_lower or 'корзина' in step_lower:
                color = '#f39c12'
            elif is_highlight:
                color = '#9b59b6'
            else:
                color = '#3498db'
            title_suffix = ""

        net.add_node(
            step,
            label=str(step),
            title=f"Шаг: {step}\\nПосещений: {step_visits.get(step, 0)}{title_suffix}",
            size=size,
            color=color,
            font={'size': 14,
                  'color': 'white' if (step in sink_nodes or step in max_outgoing_nodes or is_highlight) else 'black'}
        )

    # Добавляем рёбра
    edge_weights = edges.groupby(['step', 'next_step'])['count'].sum()
    max_weight = edge_weights.max() if not edge_weights.empty else 0

    for _, row in edges.iterrows():
        src, tgt, cnt, is_cycle = row['step'], row['next_step'], row['count'], row['is_cycle_edge']

        # Оранжевый для рёбер с максимальным весом
        if cnt == max_weight:
            color = '#f39c12'
            width = max(2.5, min(15, cnt / 2))
            title_suffix = f" 🔥 МАКС. ВЕС ({cnt})"
        elif is_cycle:
            color = '#e67e22'
            width = max(1.5, min(12, cnt / 2))
            title_suffix = " ⭮ ЦИКЛ"
        else:
            color = '#3498db'
            width = max(1.5, min(12, cnt / 2))
            title_suffix = ""

        dashes = [5, 5] if is_cycle else False

        net.add_edge(
            str(src),
            str(tgt),
            value=cnt,
            title=f"{src} → {tgt}\\nПереходов: {cnt}{title_suffix}",
            width=width,
            color=color,
            dashes=dashes,
            smooth={'type': 'curvedCW', 'roundness': 0.4}
        )

    # Настройки физики
    net.toggle_physics(False)
    #net.show_buttons(filter_=['nodes'])

    # Генерируем HTML
    try:
        html_content = net.generate_html(notebook=False)
        return html_content, f"Граф построен: {len(all_steps)} узлов, {len(edges)} рёбер"
    except Exception as e:
        return None, f"Ошибка генерации графа: {str(e)}"

def calculate_metrics(df, transitions):
    """Расчёт метрик для дашборда"""
    metrics = {
        'total_users': df['user_id'].nunique(),
        'total_events': len(df),
        'total_transitions': len(transitions),
        'unique_steps': df['step'].nunique(),
        'avg_steps_per_user': df.groupby('user_id').size().mean(),
        'cycle_rate': (transitions['is_cycle'].sum() / len(transitions) * 100) if len(transitions) > 0 else 0,
        'top_exit_step': df.groupby('user_id').last()['step'].value_counts().index[0] if not df.empty else "N/A",
        'conversion_rate': (df.groupby('user_id')['step'].apply(
            lambda x: x.str.lower().isin(['purchase', 'success', 'complete', 'оплата', 'завершение']).any()
        ).sum() / df['user_id'].nunique() * 100) if not df.empty else 0
    }
    return metrics

# =============== ОСНОВНОЙ ИНТЕРФЕЙС ===============

# Боковая панель
with st.sidebar:
    st.markdown("# Графин")
    st.header("📁 Загрузка данных")

    uploaded_file = st.file_uploader(
        "Загрузите CSV файл",
        type=["csv"],
        help="Файл должен содержать колонки: user_id, step, timestamp"
    )


# Основная область
if uploaded_file is None:
    # Страница приветствия
    st.info("""
    **Формат данных:**
    ```
    user_id, step, timestamp
    1, landing, 2024-01-01 10:00:00
    1, catalog, 2024-01-01 10:02:00
    2, landing, 2024-01-01 10:05:00
    ```
    """)

    # Пример данных для скачивания
    #st.markdown("### 📥 Пример данных с циклами")


    sample_data = pd.DataFrame({
        'user_id': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
        'step': ['landing', 'catalog', 'product', 'catalog', 'product', 'purchase',
                 'landing', 'search', 'search', 'exit',
                 'landing', 'catalog', 'cart', 'catalog', 'purchase',
                 'landing', 'exit', 'exit',
                 'landing', 'search', 'product', 'search', 'product'],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:02:00', '2024-01-01 10:05:00',
                      '2024-01-01 10:07:00', '2024-01-01 10:10:00', '2024-01-01 10:15:00',
                      '2024-01-01 11:00:00', '2024-01-01 11:01:00', '2024-01-01 11:03:00', '2024-01-01 11:05:00',
                      '2024-01-01 12:00:00', '2024-01-01 12:02:00', '2024-01-01 12:05:00',
                      '2024-01-01 12:08:00', '2024-01-01 12:12:00',
                      '2024-01-01 13:00:00', '2024-01-01 13:01:00', '2024-01-01 13:02:00',
                      '2024-01-01 14:00:00', '2024-01-01 14:01:00', '2024-01-01 14:03:00',
                      '2024-01-01 14:05:00', '2024-01-01 14:07:00']
    })

    with st.expander("### 📥 Пример данных с циклами"):
        st.dataframe(sample_data, width='stretch', height=200)
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="💾 Скачать CSV",
            data=csv,
            file_name="sample_user_journey_cycles.csv",
            mime="text/csv",
            width='stretch'
        )



else:
    # =============== ЗАГРУЗКА И ВАЛИДАЦИЯ ===============
    with st.spinner("Загрузка данных..."):
        df, error = load_and_validate_data(uploaded_file)

    if error != "OK":
        st.error(f"❌ {error}")
        st.stop()

    # =============== ПОДГОТОВКА ДАННЫХ ===============
    transitions = prepare_transitions(df)
    metrics = calculate_metrics(df, transitions)

    # Получаем список всех уникальных шагов для фильтра
    all_unique_steps = sorted(df['step'].unique())

    # Статистика по весам рёбер для ползунка
    if not transitions.empty:
        edge_weights = transitions.groupby(['step', 'next_step']).size()
        min_w, max_w = int(edge_weights.min()), int(edge_weights.max())
    else:
        min_w, max_w = 1, 10

    # =============== БОКОВАЯ ПАНЕЛЬ С ФИЛЬТРАМИ ===============
    with st.sidebar:

        st.subheader("🔍 Выбор шагов (нод)")

        # Множественный выбор шагов
        selected_nodes = st.multiselect(
            "Выберите шаги для отображения",
            options=all_unique_steps,
            default=None,
            placeholder="Выберите шаги или оставьте пустым",
            help="Отобразятся только выбранные шаги и связи между ними"
        )

        # Информация о выборе
        if selected_nodes:
            st.info(f"Выбрано шагов: {len(selected_nodes)}")
        else:
            st.info("Отображаются все шаги")

        st.subheader("⚖️ Вес рёбер")

        # Ползунок для веса рёбер
        weight_range = st.slider(
            "Диапазон количества переходов",
            min_value=min_w,
            max_value=max_w,
            value=(min_w, max_w),
            step=1,
            help="Минимальное и максимальное количество переходов для отображения рёбер"
        )
        min_weight, max_weight = weight_range

        # Показываем текущие значения
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Мин. вес", min_weight)
        with col2:
            st.metric("Макс. вес", max_weight)

        st.subheader("📅 Дата")
        date_range = st.date_input(
            "Диапазон дат",
            value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )

        # Применяем фильтр по дате
        if len(date_range) == 2:
            mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
            df_filtered = df[mask].copy()
            transitions_filtered = prepare_transitions(df_filtered)
        else:
            df_filtered = df.copy()
            transitions_filtered = transitions.copy()

        st.subheader("👥 Пользователи")
        all_users = sorted(df_filtered['user_id'].unique())

        # Ограничиваем количество опций для производительности
        if len(all_users) > 100:
            st.warning(f"Большое количество пользователей ({len(all_users)}). Рекомендуется фильтровать по дате.")
            user_filter = st.multiselect(
                "Выберите пользователей (показаны первые 100)",
                options=all_users[:100],
                default=None,
                help="Оставьте пустым для всех пользователей"
            )
        else:
            user_filter = st.multiselect(
                "Выберите пользователей",
                options=all_users,
                default=None,
                help="Оставьте пустым для всех пользователей"
            )

        if user_filter:
            df_filtered = df_filtered[df_filtered['user_id'].isin(user_filter)].copy()
            transitions_filtered = prepare_transitions(df_filtered)

        st.markdown("---")
        if st.button("🔄 Сбросить все фильтры", use_container_width=True):
            st.rerun()

    # =============== МЕТРИКИ ===============
    st.markdown("### 📊 Ключевые метрики")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Пользователей", f"{metrics['total_users']:,}")
    with col2:
        st.metric("📍 Уникальных шагов", metrics['unique_steps'])
    with col3:
        st.metric("🔄 Циклов", f"{metrics['cycle_rate']:.1f}%")
    with col4:
        st.metric("✅ Конверсия", f"{metrics['conversion_rate']:.1f}%")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("📝 Событий всего", f"{metrics['total_events']:,}")
    with col6:
        st.metric("➡️ Переходов", f"{metrics['total_transitions']:,}")
    with col7:
        st.metric("📊 Шагов на пользователя", f"{metrics['avg_steps_per_user']:.1f}")

    st.markdown(f"**Точка оттока:** `{metrics['top_exit_step']}`")

    # Краткая статистика по фильтрам
    if selected_nodes:
        st.caption(f"🔍 Фильтр: {len(selected_nodes)} шагов | ⚖️ Вес рёбер: {min_weight}-{max_weight}")
    else:
        st.caption(f"⚖️ Вес рёбер: {min_weight}-{max_weight}")

    st.markdown("---")

    # =============== ИНТЕРАКТИВНЫЙ ГРАФ ===============
    st.markdown("### 🎯 Интерактивный граф пути пользователя")

    with st.spinner("Построение графа..."):
        html_content, graph_status = build_graph_pyvis(
            transitions_filtered,
            min_weight=min_weight,
            max_weight=max_weight,
            selected_nodes=selected_nodes if selected_nodes else None
        )

    if html_content:
        # Сохраняем HTML во временный файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_file = f.name

        # Отображаем через iframe
        st.components.v1.html(
            open(temp_file, 'r', encoding='utf-8').read(),
            height=800,
            scrolling=True
        )

        # Кнопка скачивания
        with open(temp_file, 'r', encoding='utf-8') as f:
            st.download_button(
                label="💾 Скачать граф как HTML",
                data=f.read(),
                file_name=f"user_journey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                width='stretch'
            )

        os.unlink(temp_file)  # Удаляем временный файл
    else:
        st.warning(f"⚠️ {graph_status}")
        st.info("💡 Попробуйте изменить фильтры или выбрать другие шаги")

    st.markdown("---")

    # =============== ДОПОЛНИТЕЛЬНЫЕ ВИЗУАЛИЗАЦИИ ===============
    st.markdown("### 📈 Дополнительная аналитика")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📊 Топ шагов")
        if not df_filtered.empty:
            top_steps = df_filtered['step'].value_counts().head(10)
            fig_steps = px.bar(
                top_steps,
                orientation='h',
                labels={'value': 'Количество', 'index': 'Шаг'},
                title="Самые популярные шаги",
                color_discrete_sequence=['#3498db']
            )
            fig_steps.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            # ИСПРАВЛЕНО: передаём конфигурацию правильно
            st.plotly_chart(fig_steps, config={'width': 'stretch', 'displayModeBar': False})
        else:
            st.info("Нет данных")

    with col2:
        st.markdown("#### 🔁 Циклы")
        cycle_summary = transitions_filtered[transitions_filtered['is_cycle'] == True]
        if len(cycle_summary) > 0:
            cycle_types = cycle_summary.groupby(['step', 'next_step']).size().reset_index(name='count')
            cycle_types['cycle'] = cycle_types['step'] + ' → ' + cycle_types['next_step']
            cycle_types = cycle_types.sort_values('count', ascending=False).head(8)

            fig_cycles = px.bar(
                cycle_types,
                x='count',
                y='cycle',
                orientation='h',
                title="Топ циклических переходов",
                color_discrete_sequence=['#e67e22']
            )
            fig_cycles.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            # ИСПРАВЛЕНО: передаём конфигурацию правильно
            st.plotly_chart(fig_cycles, config={'width':'stretch', 'displayModeBar': False})
        else:
            st.info("Циклов не обнаружено")

    with col3:
        st.markdown("#### ⏱️ Время между шагами")
        if not df_filtered.empty:
            df_filtered['time_diff'] = df_filtered.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60
            time_stats = df_filtered.groupby('step')['time_diff'].agg(['mean', 'median']).dropna().head(10)
            if not time_stats.empty:
                fig_time = px.bar(
                    time_stats['mean'].sort_values(ascending=True),
                    orientation='h',
                    title="Среднее время на шаге (мин)",
                    color_discrete_sequence=['#9b59b6']
                )
                fig_time.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                # ИСПРАВЛЕНО: передаём конфигурацию правильно
                st.plotly_chart(fig_time, config={'width': 'stretch', 'displayModeBar': False})
            else:
                st.info("Недостаточно данных")
        else:
            st.info("Нет данных")

    # =============== ТАБЛИЦЫ ===============
    #st.markdown("### 📋 Данные")
    with st.expander("### 📋 Данные"):
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Исходные данные", "➡️ Переходы", "📈 Статистика", "🔄 Циклы"])

        with tab1:
            st.markdown(f"**Всего записей:** {len(df_filtered)}")
            st.dataframe(df_filtered, width='stretch', height=400)

            # Кнопка экспорта
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Скачать данные как CSV",
                data=csv,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )

        with tab2:
            if not transitions_filtered.empty:
                transitions_summary = transitions_filtered.groupby(['step', 'next_step']).agg(
                    переходов=('user_id', 'count'),
                    уникальных_пользователей=('user_id', 'nunique'),
                    циклов=('is_cycle', 'sum')
                ).reset_index().sort_values('переходов', ascending=False)
                st.markdown(f"**Всего переходов:** {len(transitions_summary)}")
                st.dataframe(transitions_summary, width='stretch', height=400)

                # Экспорт переходов
                csv = transitions_summary.to_csv(index=False)
                st.download_button(
                    label="📥 Скачать переходы как CSV",
                    data=csv,
                    file_name=f"transitions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            else:
                st.info("Нет переходов для отображения")

        with tab3:
            if not df_filtered.empty:
                user_stats = df_filtered.groupby('user_id').agg(
                    шагов=('step', 'count'),
                    уникальных_шагов=('step', 'nunique'),
                    первый_шаг=('step', 'first'),
                    последний_шаг=('step', lambda x: list(x)[-1]),
                    дата_начала=('timestamp', 'min'),
                    дата_конца=('timestamp', 'max')
                ).reset_index()
                st.markdown(f"**Всего пользователей:** {len(user_stats)}")
                st.dataframe(user_stats, width='stretch', height=400)
            else:
                st.info("Нет данных для статистики")

        with tab4:
            cycle_detailed = transitions_filtered[transitions_filtered['is_cycle'] == True]
            if not cycle_detailed.empty:
                st.markdown(f"**Обнаружено циклов:** {len(cycle_detailed)}")
                cycle_detailed_display = cycle_detailed[['user_id', 'step', 'next_step', 'timestamp']].copy()
                cycle_detailed_display.columns = ['Пользователь', 'Текущий шаг', 'Следующий шаг (цикл)', 'Время']
                st.dataframe(cycle_detailed_display, width='stretch', height=400)
            else:
                st.info("Циклов не обнаружено")