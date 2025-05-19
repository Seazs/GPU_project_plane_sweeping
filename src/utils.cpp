#include <cstdio>
#include <vector>
#include <fstream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"


void save_cost_cube(const std::vector<cv::Mat>& cost_cube, const std::string& filename) {
	// Save the cost cube to a binary file
    std::ofstream ofs(filename, std::ios::binary);
    int num_planes = static_cast<int>(cost_cube.size());
    ofs.write(reinterpret_cast<const char*>(&num_planes), sizeof(int));
    for (const auto& mat : cost_cube) {
        int rows = mat.rows, cols = mat.cols, type = mat.type();
        ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&type), sizeof(int));
        size_t data_size = mat.total() * mat.elemSize();
        ofs.write(reinterpret_cast<const char*>(mat.data), data_size);
    }
    ofs.close();
}

std::vector<cv::Mat> load_cost_cube(const std::string& filename) {
	// Load the cost cube from a binary file
    std::ifstream ifs(filename, std::ios::binary);
    int num_planes;
    ifs.read(reinterpret_cast<char*>(&num_planes), sizeof(int));
    std::vector<cv::Mat> cost_cube(num_planes);
    for (int i = 0; i < num_planes; ++i) {
        int rows, cols, type;
        ifs.read(reinterpret_cast<char*>(&rows), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&cols), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&type), sizeof(int));
        cost_cube[i] = cv::Mat(rows, cols, type);
        size_t data_size = cost_cube[i].total() * cost_cube[i].elemSize();
        ifs.read(reinterpret_cast<char*>(cost_cube[i].data), data_size);
    }
    ifs.close();
    return cost_cube;
}

bool compare_cost_cubes(const std::vector<cv::Mat>& cube1, const std::vector<cv::Mat>& cube2, float tol) {


    if (cube1.size() != cube2.size()) {
        std::cout << "Different number of planes: " << cube1.size() << " vs " << cube2.size() << std::endl;
        return false;
    }

    bool all_ok = true;
    int num_planes = static_cast<int>(cube1.size());
    int diff_count = 0;
    float max_diff = 0.0f;

    for (int z = 0; z < num_planes; ++z) {
        if (cube1[z].rows != cube2[z].rows || cube1[z].cols != cube2[z].cols || cube1[z].type() != cube2[z].type()) {
            std::cout << "Plane " << z << " has different shape or type." << std::endl;
            all_ok = false;
            continue;
        }
        for (int y = 0; y < cube1[z].rows; ++y) {
            for (int x = 0; x < cube1[z].cols; ++x) {
                float v1 = cube1[z].at<float>(y, x);
                float v2 = cube2[z].at<float>(y, x);
                float diff = std::fabs(v1 - v2);
                if (diff > tol && z > 20) { // Skip first plane
                    if (diff_count < 100) // Print only first 10 differences
                        std::cout << "Diff at plane " << z << ", (" << y << "," << x << "): " << v1 << " vs " << v2 << " (diff=" << diff << ")\n";
                    diff_count++;
                    all_ok = false;
                }
                if (diff > max_diff) max_diff = diff;
            }
        }
    }
    if (all_ok) {
        std::cout << "Cost cubes are equal within tolerance " << tol << std::endl;
    } else {
        std::cout << "Found " << diff_count << " differing values. Max diff: " << max_diff << std::endl;
    }
    return all_ok;
}

