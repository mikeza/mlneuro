:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. include:: gallery_backreferences/{{module}}.{{objname}}.examples

.. raw:: html

    <div class="clearer"></div>