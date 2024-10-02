import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS


def plot_scatters(data: pd.DataFrame, x_columns_with_titles: dict, y: str, color_column: str,
                  palette=None, use_px=True, hover_addition=None) -> None:
    columns_to_plot = list(x_columns_with_titles.keys())
    for i, column in enumerate(columns_to_plot):
        hover_data = {column: True, color_column: True}
        if hover_addition is not None:
            hover_data.update(hover_addition)
        if use_px:
            fig = px.scatter(
                data,
                x=column,
                y=y,
                color=color_column,
                hover_name=y,
                hover_data=hover_data,
                title=f"{i+1}. {x_columns_with_titles[column]}",
                color_continuous_scale=palette
            )
            fig.show()
            continue
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=data,
            x=column,
            y=y,
            hue=color_column,
            palette=palette if palette else 'Set2',
            s=100
        )
        plt.title(x_columns_with_titles[column])
        plt.xlabel(column)
        plt.ylabel(y)
        plt.yticks([])
        plt.xticks(rotation=45)
        plt.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


def plot_using_mds(data: pd.DataFrame, valuable_columns: list, n_components: int, hover_name: str,
                   color_column: str, title: str, hover_addition=None,
                   palette=None, random_state=0, use_px=True) -> None:
    mds_df = create_mds_df(data, n_components, valuable_columns, random_state)
    hover_data = {color_column: True}
    if hover_addition is not None:
        hover_data.update(hover_addition)

    if use_px:
        fig = px.scatter(
            mds_df,
            x='MDS1',
            y='MDS2',
            color=color_column,
            hover_name=hover_name,
            hover_data=hover_data,
            title=title,
            color_continuous_scale=palette
        )

        fig.show()
        return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=mds_df,
        x='MDS1',
        y='MDS2',
        hue=color_column,
        palette=palette if palette else 'Set2',
        s=100
    )
    plt.title(title)
    plt.xlabel('MDS1')
    plt.ylabel('MDS2')
    plt.yticks([])
    plt.xticks(rotation=45)
    plt.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def create_mds_df(data: pd.DataFrame, n_components: int, valuable_columns: list,
                  random_state=0) -> pd.DataFrame:
    mds = MDS(n_components=n_components, random_state=random_state, normalized_stress='auto')
    mds_transformed = mds.fit_transform(data[valuable_columns])
    mds_df = pd.DataFrame(mds_transformed, columns=[f'MDS{i + 1}' for i in range(n_components)])
    mds_df = pd.concat([data, mds_df], axis=1)
    return mds_df
