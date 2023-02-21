from pathlib import Path

from persona_consistency.plots import MetricsPlotter


def main():
    plotter = MetricsPlotter(
        task_names=[
            'subscribes-to-deontology',
            'conscientiousness',
        ],
        model_names=[
            'davinci',
            'davinci-instruct-beta',
            'text-davinci-001',
            'text-davinci-002',
            'text-davinci-003'
        ],
        data_path=Path('results') / '100_samples'
    )
    plotter.plot_all()

    plotter = MetricsPlotter(
        task_names=[
            'politically-conservative',
            'abortion-should-be-illegal',
            'conscientiousness',
            'neuroticism',
            'subscribes-to-deontology',
            'subscribes-to-Buddhism',
            'myopic-reward',
            'survival-instinct',
            'self-awareness-general-ai',
        ],
        model_names=[
            'davinci',
            'davinci-instruct-beta',
            'text-davinci-001',
            'text-davinci-002',
            'text-davinci-003'
        ],
        data_path=Path('results') / '20_samples'
    )
    plotter.plot_all()

if __name__ == '__main__':
    main()
