from heapq import heappop, heappush, heapify
from collections import defaultdict, OrderedDict
from bitarray import bitarray
import numpy as np

# defualt huffman table
DCLuminanceCodes = defaultdict(lambda: str, {
    0: '00',              
    1: '010',            
    2: '011',            
    3: '100',            
    4: '101',            
    5: '110',            
    6: '1110',          
    7: '11110',        
    8: '111110',      
    9: '1111110',    
    10: '11111110',  
    11: '111111110'
})

DCChrominanceCodes = defaultdict(lambda: str, {
    0: '00',                 
    1: '01',                 
    2: '10',                 
    3: '110',               
    4: '1110',             
    5: '11110',           
    6: '111110',         
    7: '1111110',       
    8: '11111110',     
    9: '111111110',   
    10: '1111111110', 
    11: '11111111110'
})

ACLuminanceCodes = defaultdict(lambda: str, {
    1: '00',
    2: '01',
    3: '100',
    0: '1010',
    4: '1011',
    17: '1100',
    5: '11010',
    18: '11011',
    33: '11100',
    49: '111010',
    65: '111011',
    6: '1111000',
    19: '1111001',
    81: '1111010',
    97: '1111011',
    7: '11111000',
    34: '11111001',
    113: '11111010',
    20: '111110110',
    50: '111110111',
    129: '111111000',
    145: '111111001',
    161: '111111010',
    8: '1111110110',
    35: '1111110111',
    66: '1111111000',
    177: '1111111001',
    193: '1111111010',
    21: '11111110110',
    82: '11111110111',
    209: '11111111000',
    240: '11111111001',
    36: '111111110100',
    51: '111111110101',
    98: '111111110110',
    114: '111111110111',
    130: '111111111000000',
    9: '1111111110000010',
    10: '1111111110000011',
    22: '1111111110000100',
    23: '1111111110000101',
    24: '1111111110000110',
    25: '1111111110000111',
    26: '1111111110001000',
    37: '1111111110001001',
    38: '1111111110001010',
    39: '1111111110001011',
    40: '1111111110001100',
    41: '1111111110001101',
    42: '1111111110001110',
    52: '1111111110001111',
    53: '1111111110010000',
    54: '1111111110010001',
    55: '1111111110010010',
    56: '1111111110010011',
    57: '1111111110010100',
    58: '1111111110010101',
    67: '1111111110010110',
    68: '1111111110010111',
    69: '1111111110011000',
    70: '1111111110011001',
    71: '1111111110011010',
    72: '1111111110011011',
    73: '1111111110011100',
    74: '1111111110011101',
    83: '1111111110011110',
    84: '1111111110011111',
    85: '1111111110100000',
    86: '1111111110100001',
    87: '1111111110100010',
    88: '1111111110100011',
    89: '1111111110100100',
    90: '1111111110100101',
    99: '1111111110100110',
    100: '1111111110100111',
    101: '1111111110101000',
    102: '1111111110101001',
    103: '1111111110101010',
    104: '1111111110101011',
    105: '1111111110101100',
    106: '1111111110101101',
    115: '1111111110101110',
    116: '1111111110101111',
    117: '1111111110110000',
    118: '1111111110110001',
    119: '1111111110110010',
    120: '1111111110110011',
    121: '1111111110110100',
    122: '1111111110110101',
    131: '1111111110110110',
    132: '1111111110110111',
    133: '1111111110111000',
    134: '1111111110111001',
    135: '1111111110111010',
    136: '1111111110111011',
    137: '1111111110111100',
    138: '1111111110111101',
    146: '1111111110111110',
    147: '1111111110111111',
    148: '1111111111000000',
    149: '1111111111000001',
    150: '1111111111000010',
    151: '1111111111000011',
    152: '1111111111000100',
    153: '1111111111000101',
    154: '1111111111000110',
    162: '1111111111000111',
    163: '1111111111001000',
    164: '1111111111001001',
    165: '1111111111001010',
    166: '1111111111001011',
    167: '1111111111001100',
    168: '1111111111001101',
    169: '1111111111001110',
    170: '1111111111001111',
    178: '1111111111010000',
    179: '1111111111010001',
    180: '1111111111010010',
    181: '1111111111010011',
    182: '1111111111010100',
    183: '1111111111010101',
    184: '1111111111010110',
    185: '1111111111010111',
    186: '1111111111011000',
    194: '1111111111011001',
    195: '1111111111011010',
    196: '1111111111011011',
    197: '1111111111011100',
    198: '1111111111011101',
    199: '1111111111011110',
    200: '1111111111011111',
    201: '1111111111100000',
    202: '1111111111100001',
    210: '1111111111100010',
    211: '1111111111100011',
    212: '1111111111100100',
    213: '1111111111100101',
    214: '1111111111100110',
    215: '1111111111100111',
    216: '1111111111101000',
    217: '1111111111101001',
    218: '1111111111101010',
    225: '1111111111101011',
    226: '1111111111101100',
    227: '1111111111101101',
    228: '1111111111101110',
    229: '1111111111101111',
    230: '1111111111110000',
    231: '1111111111110001',
    232: '1111111111110010',
    233: '1111111111110011',
    234: '1111111111110100',
    241: '1111111111110101',
    242: '1111111111110110',
    243: '1111111111110111',
    244: '1111111111111000',
    245: '1111111111111001',
    246: '1111111111111010',
    247: '1111111111111011',
    248: '1111111111111100',
    249: '1111111111111101',
    250: '1111111111111110'
})

ACChrominanceCodes = defaultdict(lambda: str, {
    0: '00',
    1: '01',
    2: '100',
    3: '1010',
    17: '1011',
    4: '11000',
    5: '11001',
    33: '11010',
    49: '11011',
    6: '111000',
    18: '111001',
    65: '111010',
    81: '111011',
    7: '1111000',
    97: '1111001',
    113: '1111010',
    19: '11110110',
    34: '11110111',
    50: '11111000',
    129: '11111001',
    8: '111110100',
    20: '111110101',
    66: '111110110',
    145: '111110111',
    161: '111111000',
    177: '111111001',
    193: '111111010',
    9: '1111110110',
    35: '1111110111',
    51: '1111111000',
    82: '1111111001',
    240: '1111111010',
    21: '11111110110',
    98: '11111110111',
    114: '11111111000',
    209: '11111111001',
    10: '111111110100',
    22: '111111110101',
    36: '111111110110',
    52: '111111110111',
    225: '11111111100000',
    37: '111111111000010',
    241: '111111111000011',
    23: '1111111110001000',
    24: '1111111110001001',
    25: '1111111110001010',
    26: '1111111110001011',
    38: '1111111110001100',
    39: '1111111110001101',
    40: '1111111110001110',
    41: '1111111110001111',
    42: '1111111110010000',
    53: '1111111110010001',
    54: '1111111110010010',
    55: '1111111110010011',
    56: '1111111110010100',
    57: '1111111110010101',
    58: '1111111110010110',
    67: '1111111110010111',
    68: '1111111110011000',
    69: '1111111110011001',
    70: '1111111110011010',
    71: '1111111110011011',
    72: '1111111110011100',
    73: '1111111110011101',
    74: '1111111110011110',
    83: '1111111110011111',
    84: '1111111110100000',
    85: '1111111110100001',
    86: '1111111110100010',
    87: '1111111110100011',
    88: '1111111110100100',
    89: '1111111110100101',
    90: '1111111110100110',
    99: '1111111110100111',
    100: '1111111110101000',
    101: '1111111110101001',
    102: '1111111110101010',
    103: '1111111110101011',
    104: '1111111110101100',
    105: '1111111110101101',
    106: '1111111110101110',
    115: '1111111110101111',
    116: '1111111110110000',
    117: '1111111110110001',
    118: '1111111110110010',
    119: '1111111110110011',
    120: '1111111110110100',
    121: '1111111110110101',
    122: '1111111110110110',
    130: '1111111110110111',
    131: '1111111110111000',
    132: '1111111110111001',
    133: '1111111110111010',
    134: '1111111110111011',
    135: '1111111110111100',
    136: '1111111110111101',
    137: '1111111110111110',
    138: '1111111110111111',
    146: '1111111111000000',
    147: '1111111111000001',
    148: '1111111111000010',
    149: '1111111111000011',
    150: '1111111111000100',
    151: '1111111111000101',
    152: '1111111111000110',
    153: '1111111111000111',
    154: '1111111111001000',
    162: '1111111111001001',
    163: '1111111111001010',
    164: '1111111111001011',
    165: '1111111111001100',
    166: '1111111111001101',
    167: '1111111111001110',
    168: '1111111111001111',
    169: '1111111111010000',
    170: '1111111111010001',
    178: '1111111111010010',
    179: '1111111111010011',
    180: '1111111111010100',
    181: '1111111111010101',
    182: '1111111111010110',
    183: '1111111111010111',
    184: '1111111111011000',
    185: '1111111111011001',
    186: '1111111111011010',
    194: '1111111111011011',
    195: '1111111111011100',
    196: '1111111111011101',
    197: '1111111111011110',
    198: '1111111111011111',
    199: '1111111111100000',
    200: '1111111111100001',
    201: '1111111111100010',
    202: '1111111111100011',
    210: '1111111111100100',
    211: '1111111111100101',
    212: '1111111111100110',
    213: '1111111111100111',
    214: '1111111111101000',
    215: '1111111111101001',
    216: '1111111111101010',
    217: '1111111111101011',
    218: '1111111111101100',
    226: '1111111111101101',
    227: '1111111111101110',
    228: '1111111111101111',
    229: '1111111111110000',
    230: '1111111111110001',
    231: '1111111111110010',
    232: '1111111111110011',
    233: '1111111111110100',
    234: '1111111111110101',
    242: '1111111111110110',
    243: '1111111111110111',
    244: '1111111111111000',
    245: '1111111111111001',
    246: '1111111111111010',
    247: '1111111111111011',
    248: '1111111111111100',
    249: '1111111111111101',
    250: '1111111111111110'
})

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    priority_queue = [Node(char, freq) for char, freq in frequencies.items()]
    heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heappop(priority_queue)
        right = heappop(priority_queue)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heappush(priority_queue, merged)

    return priority_queue[0]

def build_codes(node, codebook: defaultdict, prefix=''):
    if node:
        if node.char is not None:
            codebook[node.char] = prefix
        build_codes(node.left, codebook, prefix + '0')
        build_codes(node.right, codebook, prefix + '1')
    return codebook

def reset_huffman_tree(node):
    if node:
        node.char = None
        node.freq = 0
        reset_huffman_tree(node.left)
        reset_huffman_tree(node.right)

class Huffman:
    def __init__(self, dcMatrix, acMatrix):
        self.dcMatrix = dcMatrix
        self.acMatrix = acMatrix

    def __huffman_encoding(self, valList, isDC=True):
        frequencies = self.__CalDCCodingCategory(valList) if isDC else self.__CalACCodeingCategory(valList)
        
        root = build_huffman_tree(frequencies)
        codes = defaultdict(str)
        build_codes(root, codes)
        sorted_codes = OrderedDict(sorted(codes.items(), key=lambda item: len(item[1])))
    
        return sorted_codes

    def __CalDCCodingCategory(self, valList):
        length_count = {}
        for value in valList:
            bit_length = int(value).bit_length()
            if bit_length in length_count:
                length_count[bit_length] += 1
            else:
                length_count[bit_length] = 1        

        return length_count

    def __CalACCodeingCategory(self, valMatrix):
        freq_dict = {}

        for block in valMatrix:
            run_length = 0

            for i in range(1, len(block)):
                val = block[i]
                
                if val == 0:
                    run_length += 1

                    if run_length > 15:
                        run_size_pair = 0xF0
                        if run_size_pair in freq_dict:
                            freq_dict[run_size_pair] += 1
                        else:
                            freq_dict[run_size_pair] = 1

                        run_length = 0

                else:
                    size = int(np.ceil(np.log2(abs(val) + 1)))
                    run_size_pair = (run_length << 4) | size

                    if run_size_pair in freq_dict:
                        freq_dict[run_size_pair] += 1
                    else:
                        freq_dict[run_size_pair] = 1

                    run_length = 0
            
            if run_length > 0:
                run_size_pair = 0
                if run_size_pair in freq_dict:
                    freq_dict[run_size_pair] += 1
                else:
                    freq_dict[run_size_pair] = 1

        return freq_dict

    def __EncodeDC(self, bitStream: bitarray, dcVal, codes):
        if dcVal == 0:
            huffman_code = codes.get(0, '')
            bitStream.extend(bitarray(huffman_code))
            return
        
        bit_length = int(dcVal).bit_length()
        huffman_code = codes.get(bit_length, '')

        if(dcVal<0):
            codeList = list(bin(dcVal)[3:])
            for i in range(len(codeList)):
                if (codeList[i] == '0'):
                    codeList[i] = 1
                else:
                    codeList[i] = 0
        else:
            codeList = list(bin(dcVal)[2:])
            for i in range(len(codeList)):
                if (codeList[i] == '0'):
                    codeList[i] = 0
                else:
                    codeList[i] = 1

        huffman_code += ''.join(map(str, codeList))
        bitStream.extend(bitarray(huffman_code))
            
    def __EncodeAC(self, bitStream: bitarray, acList, codes):
        run_length = 0
        last_non_zero_index = np.where(acList == 1)[0][-1] if 1 in acList else 0

        for i in range(0,  last_non_zero_index + 1):
            val = acList[i]
            
            if val == 0:
                run_length += 1

                if run_length > 15:
                    run_size_pair = 0xF0
                    if run_size_pair in codes:
                        bitStream.extend(codes[run_size_pair])
                    else:
                        raise ValueError(f"Not find: {run_size_pair}")

                    run_length = 0
            else:
                size = int(np.ceil(np.log2(abs(val) + 1)))
                run_size_pair = (run_length << 4) | size

                if run_size_pair in codes:
                    bitStream.extend(codes[run_size_pair])
                else:
                    raise ValueError(f"Not find: {run_size_pair}")

                if val > 0:
                    bitStream.extend(bin(val)[2:].zfill(size))
                else:
                    abs_value = abs(val)
                    binary_str = format(abs_value, f'0{size}b')
                    complement_str = ''.join('1' if bit == '0' else '0' for bit in binary_str)
                    bitStream.extend(complement_str)

                run_length = 0

        if last_non_zero_index < len(acList) - 1:
            bitStream.extend(codes[0])        

    def __padding(self, bitStream: bitarray, padding_char='1'):
        padding_size = 8 - (len(bitStream) % 8)
        bitStream.extend(padding_char * padding_size)

    def CalDCACCode(self, useDefault): 
        if useDefault:
            return HuffmanTable(DCLuminanceCodes, ACLuminanceCodes, DCChrominanceCodes, ACChrominanceCodes)

        dcLuminanceCodes = self.__huffman_encoding(self.dcMatrix[:, 0], isDC=True)
        acLuminanceCodes = self.__huffman_encoding(self.acMatrix[:, 0, :], isDC=False)

        dcChrominanceCodes = self.__huffman_encoding(self.dcMatrix[:, 1] + self.dcMatrix[:, 2], isDC=True)
        acChrominanceCodes = self.__huffman_encoding(self.acMatrix[:, 1, :] + self.acMatrix[:, 2, :], isDC=False)

        return HuffmanTable(dcLuminanceCodes, acLuminanceCodes, dcChrominanceCodes, acChrominanceCodes)

    def EncodeDCAC(self, useDefault):
        bitStream = bitarray()
        tables = self.CalDCACCode(useDefault)
        
        for i in range(self.dcMatrix.shape[0]):
            # luminance
            self.__EncodeDC(bitStream, self.dcMatrix[i, 0], tables.dcLuminanceCodes)
            self.__EncodeAC(bitStream, self.acMatrix[i, 0, :], tables.acLuminanceCodes)

            # chrominance
            self.__EncodeDC(bitStream, self.dcMatrix[i, 1], tables.dcChrominanceCodes)
            self.__EncodeAC(bitStream, self.acMatrix[i, 1, :], tables.acChrominanceCodes)
            self.__EncodeDC(bitStream, self.dcMatrix[i, 2], tables.dcChrominanceCodes)
            self.__EncodeAC(bitStream, self.acMatrix[i, 2, :], tables.acChrominanceCodes)
            
        # handle FF00
        self.__padding(bitStream)
        byteStream = bitStream.tobytes()

        i = 0
        while i < len(byteStream):
            if byteStream[i] == 0xFF:
                byteStream = byteStream[:i+1] + b'\x00' + byteStream[i+1:]
                i += 1
            i += 1

        bitStream = bitarray()
        bitStream.frombytes(byteStream)

        return bitStream, tables
    
class HuffmanTable:
    def __init__(self, dcLuminanceCodes: defaultdict, acLuminanceCodes: defaultdict, dcChrominanceCodes: defaultdict, acChrominanceCodes: defaultdict):
        self.dcLuminanceCodes = dcLuminanceCodes
        self.acLuminanceCodes = acLuminanceCodes
        self.dcChrominanceCodes = dcChrominanceCodes
        self.acChrominanceCodes = acChrominanceCodes