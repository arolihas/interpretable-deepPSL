# coding: utf-8

"""
    Peptide Match OpenAPI 2.0

    This is PeptideMatch OpenAPI.

    OpenAPI spec version: 2.0.0
    Contact: chenc@udel.edu
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


from pprint import pformat
from six import iteritems
import re


class ReportSearchParameters(object):
    """
    NOTE: This class is auto generated by the swagger code generator program.
    Do not edit the class manually.
    """


    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'taxonids': 'str',
        'swissprot': 'bool',
        'isoform': 'bool',
        'uniref100': 'bool',
        'leqi': 'bool',
        'offset': 'int',
        'size': 'int'
    }

    attribute_map = {
        'taxonids': 'taxonids',
        'swissprot': 'swissprot',
        'isoform': 'isoform',
        'uniref100': 'uniref100',
        'leqi': 'leqi',
        'offset': 'offset',
        'size': 'size'
    }

    def __init__(self, taxonids=None, swissprot=None, isoform=None, uniref100=None, leqi=None, offset=None, size=None):
        """
        ReportSearchParameters - a model defined in Swagger
        """

        self._taxonids = None
        self._swissprot = None
        self._isoform = None
        self._uniref100 = None
        self._leqi = None
        self._offset = None
        self._size = None

        if taxonids is not None:
          self.taxonids = taxonids
        if swissprot is not None:
          self.swissprot = swissprot
        if isoform is not None:
          self.isoform = isoform
        if uniref100 is not None:
          self.uniref100 = uniref100
        if leqi is not None:
          self.leqi = leqi
        if offset is not None:
          self.offset = offset
        if size is not None:
          self.size = size

    @property
    def taxonids(self):
        """
        Gets the taxonids of this ReportSearchParameters.
        NCBI taxonomy IDs.

        :return: The taxonids of this ReportSearchParameters.
        :rtype: str
        """
        return self._taxonids

    @taxonids.setter
    def taxonids(self, taxonids):
        """
        Sets the taxonids of this ReportSearchParameters.
        NCBI taxonomy IDs.

        :param taxonids: The taxonids of this ReportSearchParameters.
        :type: str
        """

        self._taxonids = taxonids

    @property
    def swissprot(self):
        """
        Gets the swissprot of this ReportSearchParameters.
        Only search SwissProt protein sequences.

        :return: The swissprot of this ReportSearchParameters.
        :rtype: bool
        """
        return self._swissprot

    @swissprot.setter
    def swissprot(self, swissprot):
        """
        Sets the swissprot of this ReportSearchParameters.
        Only search SwissProt protein sequences.

        :param swissprot: The swissprot of this ReportSearchParameters.
        :type: bool
        """

        self._swissprot = swissprot

    @property
    def isoform(self):
        """
        Gets the isoform of this ReportSearchParameters.
        Include isoforms.

        :return: The isoform of this ReportSearchParameters.
        :rtype: bool
        """
        return self._isoform

    @isoform.setter
    def isoform(self, isoform):
        """
        Sets the isoform of this ReportSearchParameters.
        Include isoforms.

        :param isoform: The isoform of this ReportSearchParameters.
        :type: bool
        """

        self._isoform = isoform

    @property
    def uniref100(self):
        """
        Gets the uniref100 of this ReportSearchParameters.
        Only search UniRef100 protein sequences.

        :return: The uniref100 of this ReportSearchParameters.
        :rtype: bool
        """
        return self._uniref100

    @uniref100.setter
    def uniref100(self, uniref100):
        """
        Sets the uniref100 of this ReportSearchParameters.
        Only search UniRef100 protein sequences.

        :param uniref100: The uniref100 of this ReportSearchParameters.
        :type: bool
        """

        self._uniref100 = uniref100

    @property
    def leqi(self):
        """
        Gets the leqi of this ReportSearchParameters.
        Treat Leucine (L) and Isoleucine (I) equivalent.

        :return: The leqi of this ReportSearchParameters.
        :rtype: bool
        """
        return self._leqi

    @leqi.setter
    def leqi(self, leqi):
        """
        Sets the leqi of this ReportSearchParameters.
        Treat Leucine (L) and Isoleucine (I) equivalent.

        :param leqi: The leqi of this ReportSearchParameters.
        :type: bool
        """

        self._leqi = leqi

    @property
    def offset(self):
        """
        Gets the offset of this ReportSearchParameters.
        Off set, page starting point, with default value 0.

        :return: The offset of this ReportSearchParameters.
        :rtype: int
        """
        return self._offset

    @offset.setter
    def offset(self, offset):
        """
        Sets the offset of this ReportSearchParameters.
        Off set, page starting point, with default value 0.

        :param offset: The offset of this ReportSearchParameters.
        :type: int
        """

        self._offset = offset

    @property
    def size(self):
        """
        Gets the size of this ReportSearchParameters.
        Page size with default value 100. When page size is -1, it returns all records and offset will be ignored.

        :return: The size of this ReportSearchParameters.
        :rtype: int
        """
        return self._size

    @size.setter
    def size(self, size):
        """
        Sets the size of this ReportSearchParameters.
        Page size with default value 100. When page size is -1, it returns all records and offset will be ignored.

        :param size: The size of this ReportSearchParameters.
        :type: int
        """

        self._size = size

    def to_dict(self):
        """
        Returns the model properties as a dict
        """
        result = {}

        for attr, _ in iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """
        Returns the string representation of the model
        """
        return pformat(self.to_dict())

    def __repr__(self):
        """
        For `print` and `pprint`
        """
        return self.to_str()

    def __eq__(self, other):
        """
        Returns true if both objects are equal
        """
        if not isinstance(other, ReportSearchParameters):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """
        Returns true if both objects are not equal
        """
        return not self == other
