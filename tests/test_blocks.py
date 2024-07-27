import unittest
from tensor_build.blocks.conv_2D_block import Conv2DBlock

class TestBlock(unittest.TestCase):
    def test_block_initialization(self):
        block = Conv2DBlock(filters=32, kernel_size=(3, 3), activation='relu')
        self.assertEqual(block.filters, 32)
        self.assertEqual(block.kernel_size, (3, 3))
        self.assertEqual(block.activation, 'relu')

if __name__ == '__main__':
    unittest.main()
