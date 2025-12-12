import streamlit as st

def header_tfm():
    st.markdown(
        """
        <div style="width: 100%; clear: both; padding-bottom: 10px;">

            <!-- Logo UOC -->
            <div style="float: left; width: 45%;">
                <img src="https://www.uoc.edu/portal/_resources/common/imatges/marca_UOC/UOC_Masterbrand.jpg"
                     style="height:60px; margin-top:4px;">
            </div>

            <!-- Información del TFM -->
            <div style="float: right; width: 55%; text-align:right;">

                <p style="margin: 0; padding-top: 4px;">
                    <strong>Títol:</strong> Anàlisi descriptiu de la classificació per tòpics dels articles 
                    en accés obert al Repositori Institucional.<br>
                    Estudi de cas de la 
                    <a href="https://repositori.urv.cat/estatic/PC0011/ca_index.html" 
                       target="_blank" style="color:#000078; text-decoration:none;">
                        Universitat Rovira i Virgili
                    </a>
                </p>

                <p style="margin: 0;"><strong>Autora:</strong> María José Sánchez Martos</p>
                <p style="margin: 0;"><strong>Ensenyament:</strong> Màster en Data Science</p>
                <p style="margin: 0;"><strong>Curs acadèmic:</strong> 2025–2026</p>
                <p style="margin: 0;"><strong>Universitat:</strong> Universitat Oberta de Catalunya (UOC)</p>
            </div>

        </div>

        <div style="width:100%; height: 20px; clear: both;"></div>
        """,
        unsafe_allow_html=True,
    )
