import unittest
import numpy as np
from experiments.cube_quarter_turn.action_mapping_candidate import full_range_targets


class CandidateMappingTests(unittest.TestCase):
    def setUp(self):
        self.lower=np.array([.05852]*4+[-.1,.026152,.081568])
        self.upper=np.array([.110387]*4+[1.75,.038389,.112138])
        self.reference=np.array([.0734469386,.0794473426,.0747759843,.0716725377,1.5538694832,.0364478064,.097522223])
    def test_neutral_and_full_endpoints(self):
        for value,expected in [(-1.,self.lower),(0.,self.reference),(1.,self.upper)]:
            np.testing.assert_allclose(full_range_targets(np.full(7,value),self.reference,self.lower,self.upper),expected)
    def test_monotone_and_bounded_for_all_channels(self):
        values=np.array([full_range_targets(np.full(7,a),self.reference,self.lower,self.upper) for a in np.linspace(-2,2,101)])
        self.assertTrue(np.all(np.diff(values,axis=0)>=-1e-14))
        self.assertTrue(np.all(values>=self.lower-1e-14));self.assertTrue(np.all(values<=self.upper+1e-14))
    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):full_range_targets(np.full(7,np.nan),self.reference,self.lower,self.upper)
        with self.assertRaises(ValueError):full_range_targets(np.zeros(7),self.reference+10,self.lower,self.upper)


if __name__=='__main__':unittest.main()
