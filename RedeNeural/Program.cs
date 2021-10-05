using System;

namespace RedeNeural
{
    class Program
    {
        static void Main(string[] args)
        {
            TreinarRedeNeural();
        }

        private static void TreinarRedeNeural()
        {
            RedeNeural redeNeural = new RedeNeural(2, 3, 1, 0.1);

            bool treinar = true;
            Random rd = new Random();
            double[,] entradas = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } };
            double[,] saidasEsperadas = { { 0 }, { 1 }, { 1 }, { 0 } };

            try {
                for (var i = 0; i < 100; i++)
                {
                    if (treinar) {
                        // Executa o treinamento 10 mil vezes e testa para ver se a rede já consegue resolver o problema
                        for (var j = 0; j < 10000; j++)
                        {
                            int index = rd.Next(0, 4);
                            redeNeural.Treinar(MatrizToDoubleByIndex(entradas, index), MatrizToDoubleByIndex(saidasEsperadas, index));
                        }

                        // Passa alguns valores para a rede
                        double[] teste1 = { 0, 0 };
                        double[] teste2 = { 1, 0 };
                        double[] saida1 = redeNeural.Predizer(teste1);
                        double[] saida2 = redeNeural.Predizer(teste2);
                    
                        // Verifica se conseguiu identificar as saídas mais aproximadas do correto com base nos valores passados
                        if (saida1[0] < 0.04 && saida2[0] > 0.98){
                            treinar = false;
                            Console.WriteLine($"Treinou, execução {i}");
                            Console.WriteLine(saida1[0]);
                            Console.WriteLine(saida2[0]);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        private static double[] MatrizToDoubleByIndex(double[,] matriz, int index)
        {
            double[] saida = new double[matriz.GetLength(1)];
            for (var j = 0; j < matriz.GetLength(1); j++)
            {
                saida[j] = matriz[index, j];
            }

            return saida;
        }

        private static void ImprimeMatriz(Matriz matriz)
        {
            for (int i = 0; i < matriz.Linha; i++)
            {
                for (int j = 0; j < matriz.Coluna; j++)
                {
                    Console.Write(matriz.Valores[i,j] + " ");
                }
                Console.WriteLine();
            }
        }
    }
}
