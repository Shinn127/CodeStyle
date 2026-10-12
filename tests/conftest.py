"""Keep small fixtures cheap and separate CPU and GPU coverage explicitly."""
import os
import pytest
import torch


def pytest_configure(config):
    # Child CLI fixtures otherwise inherit large BLAS thread pools for tiny tensors.
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        case = getattr(item,'callspec',None)
        if case is not None and case.params.get('device') == 'cuda':
            item.add_marker(pytest.mark.gpu)
