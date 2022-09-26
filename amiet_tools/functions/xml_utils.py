"""Author: Michael Markus Ackermann"""


def save_XML(fname: str, data: str):
    """Save a .xml file.

    Args:
        fname (str): file name.
        data (str): data to be saved in the xml, already formated.
    """
    with open(fname+".xml", "w", encoding="utf-8") as file:
        file.write(data)
        file.close()


def XML_mic(array: list):
    """Transforms an array of floats into a .xml formatation.

    Args:
        array (list): list of microphone locations [[X, n], [Y, n], [Z, n]].

    Returns:
        (str): string of points in a .xml format.
    """
    new_pos = '<pos\tName="Point {i}"\tx="{x}"\ty="{y}"\tz="{z}" />\n'
    return "".join(
        new_pos.format(i=i + 1, x=array[0][i], y=array[1][i], z=array[2][i])
        for i in range(len(array[0]))
    )


def XML_calib(calib_data: list, cf: int = 1):
    """Transforms a calibration array of floats into a .xml formatation.

    Args:
        calib_data (list): list of the calibration data for the microphone.

    Returns:
        str: string of calibration data in a .xml format.
    """
    new_pos = '<pos\tName="Point {i}"\tfactor="{factor}"/>\n'
    return "".join(
        new_pos.format(i=i + 1, factor=calib_data[i] / cf)
        for i in range(len(calib_data))
    )
