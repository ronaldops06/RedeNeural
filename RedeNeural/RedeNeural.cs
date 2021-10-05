using RedeNeural.exceptions;
using System;
using System.Collections.Generic;
using System.Text;

namespace RedeNeural
{
    public class RedeNeural
    {
        public int NeuroniosEntrada { get; set; }
        public int NeuroniosOculta { get; set; }
        public int NeuroniosSaida { get; set; }
        public double TaxaAprendizado { get; set; }
        
        private Matriz _camadaEntrada;
        private Matriz _camadaOculta;
        private Matriz _camadaSaida;
        private Matriz _biasEntradaOculta;
        private Matriz _biasOcultaSaida;
        private Matriz _pesosEntradaOculta;
        private Matriz _pesosOcultaSaida;

        public RedeNeural(int neuroniosEntrada, int neuroniosOculta, int neuroniosSaida, double taxaAprendizado)
        {
            NeuroniosEntrada = neuroniosEntrada;
            NeuroniosOculta = neuroniosOculta;
            NeuroniosSaida = neuroniosSaida;
            TaxaAprendizado = taxaAprendizado;

            GeraBias();
            GeraPesos();
        }

        public void GeraBias()
        {
            _biasEntradaOculta = new Matriz(NeuroniosOculta, 1);
            _biasEntradaOculta.PopulaMatriz();

            _biasOcultaSaida = new Matriz(NeuroniosSaida, 1);
            _biasOcultaSaida.PopulaMatriz();
        }

        public void GeraPesos()
        {
            _pesosEntradaOculta = new Matriz(NeuroniosOculta, NeuroniosEntrada);
            _pesosEntradaOculta.PopulaMatriz();

            _pesosOcultaSaida = new Matriz(NeuroniosSaida, NeuroniosOculta);
            _pesosOcultaSaida.PopulaMatriz();
        }

        /*
         * Neste método são aplicados os cálculos na rede, obtendo valores de entrada irá aplicar os pesos, bias e função de ativação
         * dando valores de saída.
         */
        public double[] FeedForward(double[] entrada)
        {
            _camadaEntrada = Matriz.ArrayToMatriz(entrada);

            try
            {
                // INPUT -> HIDDEN
                _camadaOculta = Matriz.Multiplica(_pesosEntradaOculta, _camadaEntrada);
                _camadaOculta = Matriz.Soma(_camadaOculta, _biasEntradaOculta);
                _camadaOculta = Matriz.Sigmoid(_camadaOculta);
                // HIDDEN->OUTPUT
                _camadaSaida = Matriz.Multiplica(_pesosOcultaSaida, _camadaOculta);
                _camadaSaida = Matriz.Soma(_camadaSaida, _biasOcultaSaida);
                _camadaSaida = Matriz.Sigmoid(_camadaSaida);
            }
            catch (MatrizException ex)
            {
                throw new Exception(ex.Message);
            }

            var saida = Matriz.MatrizToArray(_camadaSaida);

            return saida;
        }

        /*
         * Este método atenda o conceito de Backpropagation, aonde irá fazer o processo inverso na rede, atualizado os pesos de todas as camadas
         * com o erro calculado.
         */
        public void Backpropagation(double[] erro)
        {
            try {
                // OUTPUT -> HIDDEN
                // Transforma o array de saída esperada em uma matriz
                Matriz saidaErro = Matriz.ArrayToMatriz(erro);

                Matriz dsignmoidSaida = Matriz.Dsigmoid(_camadaSaida);
                Matriz ocultaTransposta = Matriz.Transpoe(_camadaOculta);
                Matriz gradientSaida = Matriz.Hadamard(dsignmoidSaida, saidaErro);
                gradientSaida = Matriz.MultiplicaEscalar(gradientSaida, TaxaAprendizado);

                // Ajusta o Bias da saída para a oculta
                _biasOcultaSaida = Matriz.Soma(_biasOcultaSaida, gradientSaida);
                // Ajusta os pesos da saída para oculta
                Matriz pesosOcultaSaidaDelta = Matriz.Multiplica(gradientSaida, ocultaTransposta);
                _pesosOcultaSaida = Matriz.Soma(_pesosOcultaSaida, pesosOcultaSaidaDelta);

                // HIDDEN -> INPUT
                Matriz pesosOcultaSaidaTransposta = Matriz.Transpoe(_pesosOcultaSaida);
                Matriz ocultaErro = Matriz.Multiplica(pesosOcultaSaidaTransposta, saidaErro);
                Matriz dsignmoidOculta = Matriz.Dsigmoid(_camadaOculta);
                Matriz entradaTransposta = Matriz.Transpoe(_camadaEntrada);
                Matriz gradientOculta = Matriz.Hadamard(dsignmoidOculta, ocultaErro);
                gradientOculta = Matriz.MultiplicaEscalar(gradientOculta, TaxaAprendizado);

                // Agusta o Bias da oculta para a entrada
                _biasEntradaOculta = Matriz.Soma(_biasEntradaOculta, gradientOculta);
                // Ajusta os pesos da oculta para entrada
                Matriz pesosEntrdaOcultaDelta = Matriz.Multiplica(gradientOculta, entradaTransposta);

                _pesosEntradaOculta = Matriz.Soma(_pesosEntradaOculta, pesosEntrdaOcultaDelta);
            }
            catch (MatrizException ex)
            {
                throw new Exception(ex.Message);
            }
        }

        /*
         * Este médodo tem a função de aplicar os valores recebidos nos pesos e bias já ajustados anteriormente pela rede.
         * Após treinar a rede, este método será utilizado para retornar os valore mais aproximados do real, com base nos valores de entrada.
         * */
        public double[] Predizer(double[] valores)
        {
            try {
                return FeedForward(valores);
            }
            catch (MatrizException ex)
            {
                throw new Exception(ex.Message);
            }
        }

        public void Treinar(double[] entrada, double[] esperado)
        {
            try {
                Matriz retorno = Matriz.ArrayToMatriz(FeedForward(entrada));
                Matriz erro = Matriz.ArrayToMatriz(esperado);
                double[] erroCalculado = Matriz.MatrizToArray(Matriz.Subtrai(erro, retorno));
                Backpropagation(erroCalculado);
            }
            catch (MatrizException ex)
            {
                throw new Exception(ex.Message);
            }
        }
    }
}
