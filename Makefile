.PHONY: all clean test

all:
	mkdir -p build && cd build && cmake .. && make

clean:
	rm -rf build

test:
	mkdir -p build && cd build && cmake -DBULID_TESTS=ON .. && make
	./build/tests/test_runner

