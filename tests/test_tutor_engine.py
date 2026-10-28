import unittest
from tutor_engine import Chunk, TfidfRetriever, chunk_text, extractive_answer, make_cloze_quiz

class TutorEngineTests(unittest.TestCase):
    def setUp(self):
        self.text = ("A crystalline solid has a long-range repeating arrangement of atoms. "
                     "An amorphous solid lacks long-range periodic order.\n\n"
                     "The body-centered cubic structure contains one atom at the body center and shared corner atoms. "
                     "The face-centered cubic structure has atoms at each face center and at corners.")
    def test_chunk_text_retains_source(self):
        chunks=chunk_text(self.text,"notes.md","Section: Structures",max_chars=120,overlap=20)
        self.assertGreaterEqual(len(chunks),2)
        self.assertTrue(all(c.source=="notes.md" for c in chunks))
        self.assertTrue(all(c.locator=="Section: Structures" for c in chunks))
    def test_retriever_ranks_relevant_chunk(self):
        chunks=[Chunk(0,"Crystalline solids have repeating atomic order.","a.pdf","Page 1"),
                Chunk(1,"A BCC unit cell contains a body-centered atom.","b.pdf","Page 2")]
        result=TfidfRetriever(chunks).search("body centered atom BCC",k=2)
        self.assertTrue(result)
        self.assertEqual(result[0][0].source,"b.pdf")
    def test_extractive_answer_includes_citation(self):
        chunk=Chunk(0,"Stress is force per unit area. Engineering strain is change in length divided by original length.","lecture.pdf","Page 3")
        self.assertIn("[S1]",extractive_answer("What is stress?",[(chunk,.55)]))
    def test_quiz_is_generated_from_passage(self):
        chunk=Chunk(0,"The face-centered cubic structure has atoms at each face center and at the corners of the unit cell.","notes.md","Section: FCC")
        quiz=make_cloze_quiz([(chunk,.8)],count=2)
        self.assertTrue(quiz)
        self.assertEqual(len(quiz[0]["options"]),4)
        self.assertIn(quiz[0]["answer"],quiz[0]["options"])

if __name__=="__main__":
    unittest.main()
