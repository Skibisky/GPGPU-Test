
// Cudafy_Test.RuneCalc
extern "C" __global__  void calc_r(int n,  int* build, int buildLen0, int buildLen1,  int* stat, int statLen0,  int* mult, int multLen0, int multLen1,  int* flat, int flatLen0, int flatLen1,  int* res, int resLen0, int resLen1);

// Cudafy_Test.RuneCalc
extern "C" __global__  void calc_r(int n,  int* build, int buildLen0, int buildLen1,  int* stat, int statLen0,  int* mult, int multLen0, int multLen1,  int* flat, int flatLen0, int flatLen1,  int* res, int resLen0, int resLen1)
{
	int num = 0;
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		num = build[(i) * buildLen1 + ( 0)];
		res[(i) * resLen1 + ( 0)] = stat[(0)] * mult[num * multLen1 + ( 0)] + flat[num * flatLen1 + ( 0)];
		res[(i) * resLen1 + ( 1)] = stat[(1)] * mult[num * multLen1 + ( 1)] + flat[num * flatLen1 + ( 1)];
		res[(i) * resLen1 + ( 2)] = stat[(2)] * mult[num * multLen1 + ( 2)] + flat[num * flatLen1 + ( 2)];
		res[(i) * resLen1 + ( 3)] = stat[(3)] * mult[num * multLen1 + ( 3)] + flat[num * flatLen1 + ( 3)];
		res[(i) * resLen1 + ( 4)] = stat[(4)] * mult[num * multLen1 + ( 4)] + flat[num * flatLen1 + ( 4)];
		res[(i) * resLen1 + ( 5)] = stat[(5)] * mult[num * multLen1 + ( 5)] + flat[num * flatLen1 + ( 5)];
		res[(i) * resLen1 + ( 6)] = stat[(6)] * mult[num * multLen1 + ( 6)] + flat[num * flatLen1 + ( 6)];
		res[(i) * resLen1 + ( 7)] = stat[(7)] * mult[num * multLen1 + ( 7)] + flat[num * flatLen1 + ( 7)];
	
		num = build[(i) * buildLen1 + ( 1)];
		res[(i) * resLen1 + ( 0)] = stat[(0)] * mult[num * multLen1 + ( 0)] + flat[num * flatLen1 + ( 0)];
		res[(i) * resLen1 + ( 1)] = stat[(1)] * mult[num * multLen1 + ( 1)] + flat[num * flatLen1 + ( 1)];
		res[(i) * resLen1 + ( 2)] = stat[(2)] * mult[num * multLen1 + ( 2)] + flat[num * flatLen1 + ( 2)];
		res[(i) * resLen1 + ( 3)] = stat[(3)] * mult[num * multLen1 + ( 3)] + flat[num * flatLen1 + ( 3)];
		res[(i) * resLen1 + ( 4)] = stat[(4)] * mult[num * multLen1 + ( 4)] + flat[num * flatLen1 + ( 4)];
		res[(i) * resLen1 + ( 5)] = stat[(5)] * mult[num * multLen1 + ( 5)] + flat[num * flatLen1 + ( 5)];
		res[(i) * resLen1 + ( 6)] = stat[(6)] * mult[num * multLen1 + ( 6)] + flat[num * flatLen1 + ( 6)];
		res[(i) * resLen1 + ( 7)] = stat[(7)] * mult[num * multLen1 + ( 7)] + flat[num * flatLen1 + ( 7)];

		num = build[(i) * buildLen1 + ( 2)];
		res[(i) * resLen1 + ( 0)] = stat[(0)] * mult[num * multLen1 + ( 0)] + flat[num * flatLen1 + ( 0)];
		res[(i) * resLen1 + ( 1)] = stat[(1)] * mult[num * multLen1 + ( 1)] + flat[num * flatLen1 + ( 1)];
		res[(i) * resLen1 + ( 2)] = stat[(2)] * mult[num * multLen1 + ( 2)] + flat[num * flatLen1 + ( 2)];
		res[(i) * resLen1 + ( 3)] = stat[(3)] * mult[num * multLen1 + ( 3)] + flat[num * flatLen1 + ( 3)];
		res[(i) * resLen1 + ( 4)] = stat[(4)] * mult[num * multLen1 + ( 4)] + flat[num * flatLen1 + ( 4)];
		res[(i) * resLen1 + ( 5)] = stat[(5)] * mult[num * multLen1 + ( 5)] + flat[num * flatLen1 + ( 5)];
		res[(i) * resLen1 + ( 6)] = stat[(6)] * mult[num * multLen1 + ( 6)] + flat[num * flatLen1 + ( 6)];
		res[(i) * resLen1 + ( 7)] = stat[(7)] * mult[num * multLen1 + ( 7)] + flat[num * flatLen1 + ( 7)];

		num = build[(i) * buildLen1 + ( 3)];
		res[(i) * resLen1 + ( 0)] = stat[(0)] * mult[num * multLen1 + ( 0)] + flat[num * flatLen1 + ( 0)];
		res[(i) * resLen1 + ( 1)] = stat[(1)] * mult[num * multLen1 + ( 1)] + flat[num * flatLen1 + ( 1)];
		res[(i) * resLen1 + ( 2)] = stat[(2)] * mult[num * multLen1 + ( 2)] + flat[num * flatLen1 + ( 2)];
		res[(i) * resLen1 + ( 3)] = stat[(3)] * mult[num * multLen1 + ( 3)] + flat[num * flatLen1 + ( 3)];
		res[(i) * resLen1 + ( 4)] = stat[(4)] * mult[num * multLen1 + ( 4)] + flat[num * flatLen1 + ( 4)];
		res[(i) * resLen1 + ( 5)] = stat[(5)] * mult[num * multLen1 + ( 5)] + flat[num * flatLen1 + ( 5)];
		res[(i) * resLen1 + ( 6)] = stat[(6)] * mult[num * multLen1 + ( 6)] + flat[num * flatLen1 + ( 6)];
		res[(i) * resLen1 + ( 7)] = stat[(7)] * mult[num * multLen1 + ( 7)] + flat[num * flatLen1 + ( 7)];

		num = build[(i) * buildLen1 + ( 4)];
		res[(i) * resLen1 + ( 0)] = stat[(0)] * mult[num * multLen1 + ( 0)] + flat[num * flatLen1 + ( 0)];
		res[(i) * resLen1 + ( 1)] = stat[(1)] * mult[num * multLen1 + ( 1)] + flat[num * flatLen1 + ( 1)];
		res[(i) * resLen1 + ( 2)] = stat[(2)] * mult[num * multLen1 + ( 2)] + flat[num * flatLen1 + ( 2)];
		res[(i) * resLen1 + ( 3)] = stat[(3)] * mult[num * multLen1 + ( 3)] + flat[num * flatLen1 + ( 3)];
		res[(i) * resLen1 + ( 4)] = stat[(4)] * mult[num * multLen1 + ( 4)] + flat[num * flatLen1 + ( 4)];
		res[(i) * resLen1 + ( 5)] = stat[(5)] * mult[num * multLen1 + ( 5)] + flat[num * flatLen1 + ( 5)];
		res[(i) * resLen1 + ( 6)] = stat[(6)] * mult[num * multLen1 + ( 6)] + flat[num * flatLen1 + ( 6)];
		res[(i) * resLen1 + ( 7)] = stat[(7)] * mult[num * multLen1 + ( 7)] + flat[num * flatLen1 + ( 7)];

		num = build[(i) * buildLen1 + ( 5)];
		res[(i) * resLen1 + ( 0)] = stat[(0)] * mult[num * multLen1 + ( 0)] + flat[num * flatLen1 + ( 0)];
		res[(i) * resLen1 + ( 1)] = stat[(1)] * mult[num * multLen1 + ( 1)] + flat[num * flatLen1 + ( 1)];
		res[(i) * resLen1 + ( 2)] = stat[(2)] * mult[num * multLen1 + ( 2)] + flat[num * flatLen1 + ( 2)];
		res[(i) * resLen1 + ( 3)] = stat[(3)] * mult[num * multLen1 + ( 3)] + flat[num * flatLen1 + ( 3)];
		res[(i) * resLen1 + ( 4)] = stat[(4)] * mult[num * multLen1 + ( 4)] + flat[num * flatLen1 + ( 4)];
		res[(i) * resLen1 + ( 5)] = stat[(5)] * mult[num * multLen1 + ( 5)] + flat[num * flatLen1 + ( 5)];
		res[(i) * resLen1 + ( 6)] = stat[(6)] * mult[num * multLen1 + ( 6)] + flat[num * flatLen1 + ( 6)];
		res[(i) * resLen1 + ( 7)] = stat[(7)] * mult[num * multLen1 + ( 7)] + flat[num * flatLen1 + ( 7)];
	}
}
