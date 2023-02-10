from pathlib import Path

from persona_consistency.plots import MetricsPlotter


def main():
    plotter = MetricsPlotter(
        test_names=['subscribes-to-deontology', 'subscribes-to-Buddhism', 'conscientiousness', 'neuroticism', 'myopic-reward', 'survival-instinct', 'politically-conservative', 'believes-abortion-should-be-illegal'],
        model_names=['davinci', 'davinci-instruct-beta', 'text-davinci-001', 'text-davinci-002', 'text-davinci-003'],
        data_path=Path('results')
    )
    plotter.plot_all()

if __name__ == '__main__':
    main()
