import logging

from libs.benchmark import Benchmark

log = logging.getLogger(__name__)


@Benchmark("main")
def the_main():
    name = "Hello, World!"
    log.debug(f'Hi, {name}')


if __name__ == '__main__':
    the_main()
