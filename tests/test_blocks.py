# test_blocks.py
import unittest
from blocky.blocks.blocks import Block

class TestBlock(unittest.TestCase):
    def test_block_initialization(self):
        block = Block('Dense', {'units': 64, 'activation': 'relu'})
        self.assertEqual(block.name, 'Dense')
        self.assertEqual(block.parameters, {'units': 64, 'activation': 'relu'})

if __name__ == '__main__':
    unittest.main()
