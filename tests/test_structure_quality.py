import unittest
import numpy as np
from scripts.openfold_structure_quality import kabsch_rmsd,ca_lddt

class StructureQualityTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[0.,0,0],[2,0,0],[0,3,0],[0,0,4],[3,4,1]])
    def test_rigid_transform_removed(self):
        r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        self.assertLess(kabsch_rmsd(self.p@r+np.array([10,-4,7]),self.p),1e-12)
        self.assertEqual(ca_lddt(self.p@r+np.array([10,-4,7]),self.p)[0],1.)
    def test_reflection_not_removed(self):
        p=self.p.copy();p[:,0]*=-1
        self.assertGreater(kabsch_rmsd(p,self.p),.1)
    def test_distortion_reduces_lddt(self):
        p=self.p.copy();p[0]+=20
        self.assertLess(ca_lddt(p,self.p)[0],1)
    def test_no_neighbors_rejected(self):
        with self.assertRaisesRegex(ValueError,'No reference'):
            ca_lddt(self.p*100,self.p*100)
    def test_nonfinite_rejected(self):
        p=self.p.copy();p[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):kabsch_rmsd(p,self.p)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):ca_lddt(p,self.p)

class FeatureContractTests(unittest.TestCase):
    def test_feature_hash_checks_values_shapes_and_dtypes(self):
        import torch
        from models.openfold_adapter import tensor_digest
        a={'x':torch.arange(4,dtype=torch.float32).reshape(2,2)}
        self.assertEqual(tensor_digest(a),tensor_digest({'x':a['x'].clone()}))
        self.assertNotEqual(tensor_digest(a),tensor_digest({'x':a['x'].reshape(4)}))
        self.assertNotEqual(tensor_digest(a),tensor_digest({'x':a['x'].double()}))
        b=a['x'].clone();b[1,1]+=1
        self.assertNotEqual(tensor_digest(a),tensor_digest({'x':b}))
