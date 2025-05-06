"""
单纯加载物体或手的时候不需要转，isaaclab提供了urdf的类型接口
名称为1UrdfFileCfg
"""

"""URDF file to spawn asset from.

It uses the :class:`UrdfConverter` class to create a USD file from URDF and spawns the imported
USD file. Similar to the :class:`UsdFileCfg`, the generated USD file can be modified by specifying
the respective properties in the configuration class.

See :meth:`spawn_from_urdf` for more information.

.. note::
    The configuration parameters include various properties. If not `None`, these properties
    are modified on the spawned prim in a nested manner.

    If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
    This is done by calling the respective function with the specified properties.

"""