/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package diplom.request.signclassification;

import java.awt.Dimension;
import java.awt.Rectangle;
import javax.swing.JPanel;
import javax.swing.Scrollable;

/**
 *
 * @author dandreev
 */
public class FlowPanel extends JPanel implements Scrollable {

    private static final long serialVersionUID = -8097302158317368704L;

    public FlowPanel() {
        setLayout(new WrapLayout(WrapLayout.LEFT, 5, 5));
    }

    @Override
    public Dimension getPreferredScrollableViewportSize() {
        return super.getPreferredSize();
    }

    @Override
    public int getScrollableUnitIncrement(Rectangle visibleRect, int orientation, int direction) {
        return 64;
    }

    @Override
    public int getScrollableBlockIncrement(Rectangle visibleRect, int orientation, int direction) {
        return 64;
    }

    @Override
    public boolean getScrollableTracksViewportWidth() {
        return true;
    }

    @Override
    public boolean getScrollableTracksViewportHeight() {
        return false;
    }
}
