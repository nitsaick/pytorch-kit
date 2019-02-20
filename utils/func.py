import os

def recursive_glob(root='.', suffix=''):
    """Performs recursive glob with given suffix and root
        :param root is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(root)
            for filename in filenames if filename.endswith(suffix)]
