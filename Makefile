pkg/optimal_wordle.js: pkg/optimal_wordle_lib.js
	cat $^ worker.js > $@

pkg/optimal_wordle_lib.js: src/*.rs
	time wasm-pack build --release --target no-modules
	mv pkg/optimal_wordle.js $@
