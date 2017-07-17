//
// Created by marscher on 7/17/17.
//

#ifndef PYEMMA_REGSPACE_H
#define PYEMMA_REGSPACE_H


#include <Clustering.h>


template<typename dtype>
class RegularSpaceClustering : public ClusteringBase<dtype> {
public:
    RegularSpaceClustering(dtype dmin, std::size_t max_clusters,
                           const std::string& metric,
                           size_t input_dimension) :
            ClusteringBase<dtype>(metric, input_dimension),
            dmin(dmin),
            max_clusters(max_clusters) {}

    /**
     * loops over all points in chunk and checks for each center if the distance is smaller than dmin,
     * if so, the point is appended to py_centers. This is done until max_centers is reached or all points have been
     * added to the list.
     * @param chunk array shape(n, d)
     * @param py_centers python list containing found centers.
     */
    void cluster(py::array_t <dtype, py::array::c_style> chunk, py::list py_centers) {
        std::size_t N_frames = chunk.shape(0);
        std::size_t dim = chunk.shape(1);
        std::size_t N_centers = py_centers.size();
        const dtype *data = chunk.data();
        /* do the clustering */
        for (std::size_t i = 0; i < N_frames; ++i) {
            dtype mindist = std::numeric_limits<dtype>::max();

            for (std::size_t j = 0; j < N_centers; ++j) {
                //dtype d = metric->compute(&data[i * dim], py_centers[j].data());
                dtype d = 1;
                if (d < mindist) mindist = d;
            }
            if (mindist > dmin) {
                if (N_centers + 1 > max_clusters) {
                    // TODO: custom exception type to support message?
                    throw std::exception();//"Maximum number of cluster centers reached. Consider increasing max_clusters or choose a larger minimum distance, dmin.");
                }

                // add newly found center
                py_centers.append(chunk.at(i)); //py::array_t<dtype>(sizeof(dtype) * dim, &data[i * dim]));
                N_centers++;
            }
        }
    }

protected:
    dtype dmin;
    std::size_t max_clusters;

};

#endif //PYEMMA_REGSPACE_H
