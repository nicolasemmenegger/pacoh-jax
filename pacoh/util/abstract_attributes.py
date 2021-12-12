from abc import ABCMeta

def abstractattribute(func):
    func.__is_abstract_attribute__ = True
    return func


class AbstractAttributesABCMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        # taken from https://stackoverflow.com/questions/23831510/abstract-attribute-not-property
        # the call method is called when a class constructor is called and checks whether the constructor
        # this is quite restrictive, as it rules out some other ways to construct classes, i.e. through classmethods
        # using a shared but incomplete __init__. However, for the current purposes of this codebase, it is fine for
        # now
        instance = ABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), '__is_abstract_attribute__', False)
        }
        if abstract_attributes:
            raise NotImplementedError(
                "Can't instantiate abstract class {} with"
                " abstract attributes: {}".format(
                    cls.__name__,
                    ', '.join(abstract_attributes)
                )
            )
        return instance
