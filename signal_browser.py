# based on https://github.com/pyqtgraph/pyqtgraph/blob/master/examples/hdf5.py

import numpy as np
import h5py
import pyqtgraph as pg


pg.mkQApp()
plt = pg.plot()
pg.setConfigOptions(useOpenGL=True, antialias=True)
plt.setWindowTitle("Signal Browser")
plt.enableAutoRange(False, False)
plt.setXRange(0, 1000)


class DataSource:
    @staticmethod
    def _min_max_from_min_max(min_in, max_in, ds):
        tmp_min = min_in[:(len(min_in) // ds) * ds].reshape(len(min_in) // ds, ds).min(axis=1)
        tmp_max = max_in[:(len(max_in) // ds) * ds].reshape(len(max_in) // ds, ds).max(axis=1)
        return tmp_min, tmp_max

    @staticmethod
    def _min_max_from_hdf5(hdf5, start, stop, ds, chunk_size):
        start = max(0, start)
        stop = min(len(hdf5), stop)

        ds = max(1, ds)

        samples = 1 + ((stop - start) // ds)
        tmp_min = np.zeros(samples, dtype=hdf5.dtype)
        tmp_max = np.zeros(samples, dtype=hdf5.dtype)

        chunk_size = (chunk_size // ds) * ds

        source_idx = start
        target_idx = 0

        while source_idx < stop - 1:
            chunk = hdf5[source_idx:min(stop, source_idx + chunk_size)]
            source_idx += len(chunk)

            chunk_min, chunk_max = DataSource._min_max_from_min_max(chunk, chunk, ds)

            tmp_min[target_idx:target_idx + len(chunk_min)] = chunk_min
            tmp_max[target_idx:target_idx + len(chunk_max)] = chunk_max

            target_idx += len(chunk_min)

        return tmp_min, tmp_max

    def __init__(self, hdf5, ds_from=64, ds_to=65536, offset=0):
        self._hdf5 = hdf5
        self._chunk_size = 2 ** 20

        self._ds_from = ds_from
        self._ds_to = ds_to
        self._offset = offset

        self._buffer_y = {}
        self._buffer_x = {}

        ds = self._ds_from

        tmp_min, tmp_max = DataSource._min_max_from_hdf5(self._hdf5, 0, len(self._hdf5), ds, self._chunk_size)

        tmp_min += self._offset
        tmp_max += self._offset

        tmp_y = np.zeros(len(tmp_min) + len(tmp_max), dtype=self._hdf5.dtype)
        tmp_y[0::2] = tmp_min
        tmp_y[1::2] = tmp_max
        tmp_x = np.arange(len(tmp_y)) * ds / 2

        self._buffer_x[ds] = tmp_x
        self._buffer_y[ds] = tmp_y
        print(f"Downsampled to 1:{ds} ({len(self._hdf5)} to {len(tmp_min)} samples)")

        while ds < self._ds_to:
            tmp_min, tmp_max = DataSource._min_max_from_min_max(tmp_min, tmp_max, 2)
            tmp_y = np.zeros(len(tmp_min) + len(tmp_max), dtype=self._hdf5.dtype)
            tmp_y[0::2] = tmp_min
            tmp_y[1::2] = tmp_max

            ds *= 2
            tmp_x = np.arange(len(tmp_y)) * ds / 2
            self._buffer_x[ds] = tmp_x
            self._buffer_y[ds] = tmp_y
            print(f"Downsampled to 1:{ds} ({len(tmp_min)*2} to {len(tmp_min)} samples)")

    def data(self, start, stop, down_sampling):
        start = max(0, start)
        stop = min(len(self._hdf5), stop)

        if down_sampling <= 1:
            y = self._hdf5[start:stop] + self._offset
            x = np.arange(0, len(y))
            print(f"No downsampling - read {len(y)} samples from file")
        elif down_sampling < self._ds_from:
            ds = int(down_sampling+0.5)
            tmp_min, tmp_max = self._min_max_from_hdf5(self._hdf5, start, stop,
                                                       ds, self._chunk_size)
            y = np.zeros(len(tmp_min) + len(tmp_max), dtype=self._hdf5.dtype)
            y[0::2] = tmp_min + self._offset
            y[1::2] = tmp_max + self._offset
            x = np.arange(0, len(y)) * ds / 2
            print(f"Downsampling 1:{ds} - read {len(y)} samples from file")
        else:
            ds = self._ds_from
            while ds * 2 <= down_sampling and ds * 2 <= self._ds_to:
                ds *= 2

            start_idx = start * 2 // ds
            stop_id = stop * 2 // ds
            y = self._buffer_y[ds][start_idx:stop_id]
            x = self._buffer_x[ds][:(stop_id-start_idx)]
            print(f"Downsampling 1:{ds} - read {len(y)} samples from buffer")

        return x, y


class HDF5Plot(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self.source = None
        pg.PlotCurveItem.__init__(self, *args, **kwds)

    def setSource(self, source):
        self.source = source
        self.updateHDF5Plot()

    def viewRangeChanged(self):
        self.updateHDF5Plot()

    def updateHDF5Plot(self):
        if self.source is None:
            self.setData([])
            return

        vb = self.getViewBox()
        if vb is None:
            return  # no ViewBox yet

        # Determine what data range must be read from HDF5
        xrange = vb.viewRange()[0]
        start = max(0, int(xrange[0] - 1))
        stop = int(xrange[1] + 2)

        ds = (xrange[1] - xrange[0]) // 2000  # TODO: assume window is 2000 px wide

        x, y = self.source.data(start, stop, ds)

        self.setData(x, y)  # update the plot
        self.setPos(start, 0)  # shift to match starting index
        self.resetTransform()
        self.scale()


def createFile(size):
    """Create a large HDF5 data file for testing."""
    chunk = np.random.normal(size=1000000).astype(np.float32)

    f = h5py.File("test.hdf5", "w")
    f.create_dataset("data", data=chunk, chunks=True, maxshape=(None,))
    data = f["data"]

    nChunks = size // (chunk.size * chunk.itemsize)
    for i in range(nChunks):
        newshape = [data.shape[0] + chunk.shape[0]]
        data.resize(newshape)
        data[-chunk.shape[0]:] = chunk
    f.close()


n_chans = 64
createFile(int(2 * 1e9 / n_chans))
f = h5py.File("test.hdf5", "r")

for offset in range(0, n_chans * 6, 6):
    s = DataSource(f["data"], offset=offset)
    curve = HDF5Plot()
    curve.setSource(s)
    plt.addItem(curve)

if __name__ == "__main__":
    pg.exec()
