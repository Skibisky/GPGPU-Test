
// Cudafy_Test.RuneCalc
extern "C" __global__  void calc_r(int n,  int* build, int buildLen0, int buildLen1,  int* stat, int statLen0,  int* mult, int multLen0, int multLen1,  int* flat, int flatLen0, int flatLen1,  int* res, int resLen0, int resLen1);

// Cudafy_Test.RuneCalc
extern "C" __global__  void calc_r(int n,  int* build, int buildLen0, int buildLen1,  int* stat, int statLen0,  int* mult, int multLen0, int multLen1,  int* flat, int flatLen0, int flatLen1,  int* res, int resLen0, int resLen1)
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
	{
		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 6; k++)
			{
				int num = build[(i) * buildLen1 + ( k)];
				res[(i) * resLen1 + ( j)] = stat[(j)] * mult[(num) * multLen1 + ( j)] + flat[(num) * flatLen1 + ( j)];
			}
		}
	}
}
