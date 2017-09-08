#include "Common.h"

namespace TmmModel {

	template<typename T> T Interpolate(double x, const Eigen::ArrayXd & xs, const Eigen::Array<T, Eigen::Dynamic, 1> & ys) {
		// xs must be sorted

		// Check range
		if (x < xs(0) || x >= xs(xs.size() - 1)) {
			throw std::runtime_error("Interpolation out of range");
		}

		if (xs(0) >= xs(xs.size() - 1)) {
			throw std::runtime_error("Interpolation: xs must be sorted");
		}

		// Binary search (last element, that is less or equal than x)
		int b = 0, e = xs.size() - 1;
		while (b < e) {
			int a = (b + e) / 2;
			if (xs(b) >= xs(e)) {
				throw std::runtime_error("Interpolation: xs must be sorted");
			}

			if (xs(a) > x) {
				// [b..a[
				e = a - 1;

				if (xs(e) <= x) {
					b = e;
				}
			}
			else {
				// [a..e]
				b = a;
				if (xs(a + 1) > x) {
					e = a;
				}
			}
		}
		// Linear interpolation in range x[b]..x[b+1]
		double dx = xs(b + 1) - xs(b);
		T dy = ys(b + 1) - ys(b);
		T res = ys(b) + dy / dx * (x - xs(b));
		return res;
	}

	template dcomplex TmmModel::Interpolate<dcomplex>(double, const ArrayXd &, const ArrayXcd &);
	template double TmmModel::Interpolate<double>(double, const ArrayXd &, const ArrayXd &);
}